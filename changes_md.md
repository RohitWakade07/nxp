# Speed Configuration Changes

We updated the default speeds for the rover inside the `NavigationAction.tick` method to run at 70% of the maximum throttle rather than the previous 50%. This helps the buggy move faster through the track while maintaining stable slow-down behavior for sharp curves.

### Specific Changes Made in `b3rb_ros_line_follower.py`

- **Normal Speed Increased:** 
  The nominal speed when the line is clearly detected was increased from `0.5` to `0.7`.
  ```python
  target_speed = 0.7  # Normal speed set to 70%
  ```

- **Intersection Speed Increased:** 
  When the rover loses the lines and relies on its blind intersection logic (`vector_count == 0`), the minimum target speed was increased from `0.2` to `0.4`.
  ```python
  if vectors.vector_count == 0:
      target_speed = 0.4
  ```

- **Speed on Curves Adjusted:** 
  The dynamic speed calculation that slows the rover down depending on tracking error was adjusted to start from a higher base speed of `0.7`. The minimum floor speed was also bumped to `0.3` to keep momentum up.
  ```python
  # For 1-line tracking
  target_speed = max(0.3, 0.7 - abs(error) * 0.4)
  
  # For 2-line tracking
  target_speed = max(0.3, 0.7 - abs(error) * 0.5)
  ```
