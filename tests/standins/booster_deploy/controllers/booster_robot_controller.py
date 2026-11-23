"""Stand-in for booster_deploy.controllers.booster_robot_controller, never imported (the vendor's needs ROS 2): only
the line deploy/booster_t2's exit hook depends on, as the vendor's BoosterRobotPortal.run() has it.

        exit_mode = "damping" if self._safety_abort else self.cfg.booster.exit_mode.strip().lower()
"""
