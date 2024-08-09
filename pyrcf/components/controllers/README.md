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
  - `IKReferenceInterpolator`
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
  - `IKReferenceInterpolator`
  - `BlindForwardingPlanner` (if applicable)
