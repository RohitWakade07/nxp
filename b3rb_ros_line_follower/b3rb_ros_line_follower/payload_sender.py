#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from synapse_msgs.msg import ServerCommunication
import threading
import tkinter as tk
from tkinter import ttk

class PayloadSenderGUI(Node):
    def __init__(self):
        super().__init__('payload_sender_gui')
        
        # Publisher to buggy
        self.publisher = self.create_publisher(ServerCommunication, '/ServerCommunication', 10)
        
        # Subscriber to listen to buggy messages (to get UID for ACKs)
        self.subscription = self.create_subscription(
            ServerCommunication,
            '/ServerCommunication',
            self.listener_callback,
            10
        )
        
        self.uid_counter = 0
        self.last_received_uid = None
        self.last_received_msg = None
        self.gui_update_callback = None

    def listener_callback(self, msg):
        # Buggy sends messages with src=1, dest=2
        if msg.src == 1 and msg.dest == 2:
            self.last_received_uid = msg.uid
            self.last_received_msg = msg.msg
            self.get_logger().info(f"Received msg from Buggy: '{msg.msg}' with UID {msg.uid}")
            if self.gui_update_callback:
                self.gui_update_callback(msg.msg, msg.uid)
            
            # Automatically send ACK back to buggy if it's an un-ACKed message (ack=0)
            if msg.ack == 0:
                self.send_ack()

    def send_target(self, target_char):
        msg = ServerCommunication()
        msg.src = 2
        msg.dest = 1
        msg.ack = 0
        self.uid_counter = (self.uid_counter + 1) % 256
        msg.uid = self.uid_counter
        msg.msg = target_char
        self.publisher.publish(msg)
        self.get_logger().info(f"Sent TARGET: '{target_char}' (uid={msg.uid})")

    def send_ack(self):
        if self.last_received_uid is None:
            self.get_logger().warn("Cannot send ACK: No message received from buggy yet!")
            return
        
        msg = ServerCommunication()
        msg.src = 2
        msg.dest = 1
        msg.ack = 1
        msg.uid = self.last_received_uid
        msg.msg = ""
        self.publisher.publish(msg)
        self.get_logger().info(f"Sent ACK for UID {msg.uid}")

def run_ros(node):
    rclpy.spin(node)

def main(args=None):
    rclpy.init(args=args)
    node = PayloadSenderGUI()
    
    # Run ROS2 spin in background thread
    ros_thread = threading.Thread(target=run_ros, args=(node,), daemon=True)
    ros_thread.start()

    # GUI Setup
    root = tk.Tk()
    root.title("Buggy Comm Center")
    root.geometry("480x360")
    root.configure(bg="#2c3e50")

    # Styling
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TButton', font=('Helvetica', 12, 'bold'), width=10)
    style.configure('Ack.TButton', font=('Helvetica', 11, 'bold'), width=14, foreground="white", background="#27ae60")
    style.configure('Park.TButton', font=('Helvetica', 11, 'bold'), width=14, foreground="white", background="#e67e22")

    # Title
    title_label = tk.Label(root, text="Buggy Communication Controller", font=("Helvetica", 14, "bold"), fg="#ecf0f1", bg="#2c3e50")
    title_label.pack(pady=10)

    # Status Monitor Frame
    status_frame = tk.LabelFrame(root, text=" Buggy Status Monitor ", font=("Helvetica", 10, "bold"), fg="#ecf0f1", bg="#34495e", bd=2, relief="groove")
    status_frame.pack(fill="x", padx=15, pady=5)

    status_text = tk.StringVar(value="Waiting for messages from buggy...")
    status_label = tk.Label(status_frame, textvariable=status_text, font=("Helvetica", 10), fg="#f1c40f", bg="#34495e", anchor="w")
    status_label.pack(fill="x", padx=10, pady=10)

    def update_gui_status(msg, uid):
        status_text.set(f"Last Msg: '{msg}'  |  UID: {uid}")

    node.gui_update_callback = update_gui_status

    # Button Grid Frame
    grid_frame = tk.Frame(root, bg="#2c3e50")
    grid_frame.pack(pady=10)

    # Targets: A, B, C, X, Y, Z
    targets = ['A', 'B', 'C', 'X', 'Y', 'Z']
    for idx, target in enumerate(targets):
        row = idx // 3
        col = idx % 3
        btn = ttk.Button(grid_frame, text=f"Send {target}", command=lambda t=target: node.send_target(t))
        btn.grid(row=row, column=col, padx=8, pady=8)

    # Action Buttons Frame (ACK and PARKING)
    action_frame = tk.Frame(root, bg="#2c3e50")
    action_frame.pack(pady=10)

    ack_btn = ttk.Button(action_frame, text="Send ACK", style='Ack.TButton', command=node.send_ack)
    ack_btn.pack(side="left", padx=5)

    park_btn = ttk.Button(action_frame, text="Send PARKING", style='Park.TButton', command=lambda: node.send_target('PARKING'))
    park_btn.pack(side="right", padx=5)

    def on_closing():
        root.destroy()
        node.destroy_node()
        rclpy.shutdown()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Run the Tkinter main loop
    root.mainloop()

if __name__ == '__main__':
    main()
