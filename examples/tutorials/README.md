# PyRCF -- Basic Concepts

LMCF follows the principle of a single thread control loop where components are communicating with each other strictly using pre-defined message types,
and run sequentially.

## Example of a generic control loop

```python

# NOTE: this is not meant to be a script that you can use directly. See it as more of a pseudocode
# for a control loop that can be written using the framework.

period: float = 1.0 / loop_rate
while True:
    # read latest robot state
    robot_state = robot.read()

    # use state estimator to compute robot states that are not directly observable from the robot interface
    # such as robot pose in the world, base velocity, foot contact states, etc.
    robot_state = state_estimator.update_robot_state_with_state_estimates(
        robot_state=robot_state
    )

    # generate global plan for local planner (here, we simply generate a fake plan using user input)
    # but in real world scenarious, this is typically a trajectory plan or action sequence for
    # achieving a specific task goal (e.g. trajectory from point A to B avoiding collision, or
    # sequence of actions/trajectories to achieve a manipulation action)
    global_plan = global_planner.generate_global_plan()

    # generate local plan/references for controller using a local planner, given global plan target
    local_plan = planner.generate_local_plan(
        robot_state=robot_state, global_plan=global_plan
    )

    # use the local plan to generate instantaneous command to be sent to the robot
    cmd = controller.update(robot_state=robot_state, local_plan=local_plan)

    # write the commands to the robot
    robot.write(cmd=cmd)

    # maintain control loop frequency (this is naive implementation)
    time.sleep(period)

```

## Components in a control loop

1. Robot interface: this is the interface talking to the robot. It exposes two main functions: read (to read latest state of the robot) and write
   (to write low-level commands to actuators of the robot).
2. State estimator: This component is used to estimate robot states that are not directly observable from the robot's sensors (e.g. global pose of the robot in the world).
3. Global planner: High-level planner (this could be user instructions such as joystick commands or a component that can generate global trajectories for the robot to follow).
4. Local planner: This is mostly a decoupling of what typically could be seen as part of 'controller'. See 'Agent' definition in 'additional components' section below.
   Local planner is treated an intermediary between the global planner and controller in case this is needed. This can be useful in case of legged robots for instance,
   where there is a need to have specific gait plans etc to be able to follow a global plan.
5. Controller: This component computes the low-level (joint) commands to be sent to the robot so as to follow the output of the local planner.

### Additional components

1. Agent: In a classical learning-based setup, an agent is an entity that observes the state of the robot and responds with an action in order to achieve an objective.
   With this in mind, an agent for us combines the functions of a 'local planner' and 'controller' into a single entity called 'Agent', which observes the state of the
   robot interface and responds by sending commands to the robot, so as to follow the objective of following the 'global plan' from the global planner.
2. Controller manager: controller manager is used if there are multiple 'agents' in the control loop, i.e. there are multiple controllers (e.g. with different objectives
   such as manipulation and locomotion). An ideal controller manager's function is to coordinate these multiple controllers running in the loop (such as managing, switching,
   starting, stopping controllers).
3. Command accumulation policies: These are not necessarily separate components, but rather strategies used to handle the situation of combining commands when multiple
   controllers in the control loop are sending commands to the same joint. This can happen for instance, if a mobile manipulator can use the same joint(s) to perform
   manipulation and locomotion.

## Other Notes

### On clock, time and frequencies

1. To avoid synch issues, memory issues and other deadlock issues, control loops are typically written in one thread without parallelisation (such as the ROS control framework). LMCF follows this principle. (NOTE: Typically, planners run in a different thread, however, here we include them in the same thread for simplicity, and since the simplest planners are essentially instantaneous reference generators for controllers.)

    This way of keeping the control loop as a single sequence thread is mostly very good, and leads to clear code structure, and usually allows in-place replacement of individual components in the control loop.

    However, the down side is that all components have to run in a non-blocking way, and should respect the control loop time constraints.

    NOTE: Multi-threading is still possible in this framework if the corresponding component class has threading inside their class definition.

    The other constraint the single-thread formulation brings is that each component in the control loop can only run run at exact fractions of the main control loop rate (e.g. if the main loop is 1000 Hz, each component can run at 500, 100, 200 etc. but not at, say 300Hz because this is not a perfect factor of 1000).

2. The control loop should run as fast as the demand of the `robot.write()` method for stable control. This is the fastest loop (there are cases where the main loop can run faster than the robot I/O, but this is not common). The controller is typically the next most demanding, followed by the local planner and then the global planner (out of scope of this library).

3. Typically for ensuring all components use the same clock, control frameworks share a common clock, or pass time (and dt) in each loop to the component at every time step. In LMCF, follow this philosophy to enforce synchronisation.

### Notes on design/architecture and implementation

This section covers details that are specific to LMCF library.

- All components are defined and restricted to use specific input-output function signatures (defined in their corresponding base/interface classes), but are free to do any implementation within their individual classes.
- There are 3 cartesian coordinate frames used in this architecture:
    1. World frame: This is the absolute global and inertial frame we can use to describe direct pose values (most likely for state estimates and between planners (global and local)).
    2. Base frame: This is the frame that is fixed to and moves with the base link of the robot (for floating base robots). As of now, we only describe the cartesian velocities of the robot base (linear and angular) in this frame (following navstack and ros conventions). E.g. the value of `twist` in state estimates.
    3. Teleop frame: This is the frame that moves with the robot. It is fixed at the center of the robot's footprint in the world, i.e. the origin of this frame in the world frame is at `[x, y, 0]` (where x and y are the XY coordinates of the base frame of the robot), with roll-pitch-yaw values are `[0, 0, yaw]` where `yaw` is the absolute yaw of the base frame of the robot with respect to the global world frame. This frame is used mostly for defining teleoperation commands, which are more intuitive for the user in this frame.

    NOTE: Be mindful of the data format and coordinate frame requirements of/for each component.

- The simple control loop architecture assumes all components return quick enough for the control loop to maintain the required frequency. If your component requires heavier compute, be aware that this can slow down the control loop and performance may degrade. If required, do appropriate parallelisation within the component (e.g. creating a separate thread/process within the component class).
