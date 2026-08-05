# Controllers

Controllers get the current state of the robot via the `RobotState` object and the output of the local planner (`LocalMotionPlan`), and outputs the `RobotCmd` to be sent to the robot's joints (actuators).

## Available controllers

### `JointPDController`

Joint PD controller for joint position and velocity tracking.

- `LocalMotionPlan` fields used by controller:
  - `control_mode`
  - `joint_references.joint_positions`
  - `joint_references.joint_velocities`
  - `joint_references.joint_names`

- `RobotState` fields used by controller:
  - `joint_states`

- Output fields of `RobotCmd` populated by controller:
  - `Kp`, `Kd`  
  - `joint_commands.joint_positions`
  - `joint_commands.joint_velocities`
  - `joint_commands.joint_names`

- Compatible LocalPlanner Classes:
  - `JointReferenceInterpolator`
  - `PybulletIKReferenceInterpolator`
  - `BlindForwardingPlanner` (if applicable)

### `GravityCompensatedPDController`

Joint PD controller for joint position and velocity tracking with active gravity compensation.

- `LocalMotionPlan` fields used by controller:
  - `control_mode`
  - `joint_references.joint_positions`
  - `joint_references.joint_velocities`
  - `joint_references.joint_names`

- `RobotState` fields used by controller:
  - `joint_states`

- Output fields of `RobotCmd` populated by controller:
  - `Kp`, `Kd`  
  - `joint_commands.joint_positions`
  - `joint_commands.joint_velocities`
  - `joint_commands.joint_efforts`
  - `joint_commands.joint_names`

- Compatible LocalPlanner Classes:
  - `JointReferenceInterpolator`
  - `PybulletIKReferenceInterpolator`
  - `BlindForwardingPlanner` (if applicable)

### `SegwayPIDBalanceController`

Balance (and velocity tracking) controller for a 2-wheeled segway-type robot, e.g. `upkie`.
Adapted from the wheel controller in [Stéphane Caron's upkie repo](https://github.com/upkie/upkie/blob/main/pid_balancer/wheel_controller.py).

Outputs pure joint torques: wheel joints track a balancing ground velocity, while all other
joints hold the posture they were in when the controller first ran.

- `LocalMotionPlan` fields used by controller:
  - `control_mode`
  - `twist.linear` (x used as forward ground velocity target)
  - `twist.angular` (z used as yaw rate target)

- `RobotState` fields used by controller:
  - `joint_states`
  - `state_estimates.pose.orientation` (for torso pitch)
  - `state_estimates.end_effector_states.contact_states` (for takeoff detection; if not
    populated, the wheels are assumed to be on the ground)

- Output fields of `RobotCmd` populated by controller:
  - `joint_commands.joint_efforts`
  - `joint_commands.joint_names`

- Compatible LocalPlanner Classes:
  - `BlindForwardingPlanner`

- Requires a `PinocchioInterface` (used to read wheel/end-effector poses), so the robot
  interface must be created with `create_pinocchio_interface=True`.

- Raises `RuntimeError` when the torso pitch exceeds `fall_pitch`, which the control loop
  treats as a fatal error (all components are shut down cleanly before it propagates).
