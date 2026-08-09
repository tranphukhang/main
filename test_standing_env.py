import torch

from mjlab.envs import ManagerBasedRlEnv

from standing_env_cfg import standing_env_cfg


def main():

    # ---------------------------------------------------------
    # Create environment
    # ---------------------------------------------------------

    cfg = standing_env_cfg()

    env = ManagerBasedRlEnv(
        cfg=cfg,
        device="cuda:0",
    )

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    obs, extras = env.reset()

    robot = env.scene["robot"]

    print("\n========== ENV INFO ==========")

    print("num envs:")
    print(env.num_envs)

    print("\nphysics dt:")
    print(env.physics_dt)

    print("\nenvironment dt:")
    print(env.step_dt)

    print("\nmax episode length:")
    print(env.max_episode_length)

    print("\naction dim:")
    print(env.action_manager.total_action_dim)

    print("\nactor observation shape:")
    print(obs["actor"].shape)

    print("\ncritic observation shape:")
    print(obs["critic"].shape)

    print("\njoint names:")
    print(robot.joint_names)

    print("\nactuator names:")
    print(robot.actuator_names)

    print("\ninitial root position:")
    print(robot.data.root_link_pos_w[0])

    print("\ninitial projected gravity:")
    print(robot.data.projected_gravity_b[0])

    print("\ninitial joint position:")
    print(robot.data.joint_pos[0])

    print("==============================\n")

    # ---------------------------------------------------------
    # Zero action
    # ---------------------------------------------------------

    actions = torch.zeros(
        (
            env.num_envs,
            env.action_manager.total_action_dim,
        ),
        device=env.device,
    )

    # ---------------------------------------------------------
    # Rollout
    # ---------------------------------------------------------

    for step in range(200):

        obs, reward, terminated, timeout, extras = env.step(actions)

        if step % 20 == 0:

            print(f"\n---------- step {step} ----------")

            print(
                "root position:",
                robot.data.root_link_pos_w[0]
            )

            print(
                "projected gravity:",
                robot.data.projected_gravity_b[0]
            )

            print(
                "total reward:",
                reward[0].item()
            )

            print(
                "terminated:",
                terminated[0].item()
            )

            print(
                "timeout:",
                timeout[0].item()
            )

            print("reward terms:")

            for name, value in env.reward_manager.get_active_iterable_terms(0):
                print(
                    f"  {name:25s}: {value[0]: .6f}"
                )

        # Nếu robot bị termination thì dừng test.
        if terminated[0] or timeout[0]:
            print(
                f"\nEpisode ended at step {step}"
            )
            break

    # ---------------------------------------------------------
    # Final state
    # ---------------------------------------------------------

    print("\n========== FINAL STATE ==========")

    print("root position:")
    print(robot.data.root_link_pos_w[0])

    print("\nprojected gravity:")
    print(robot.data.projected_gravity_b[0])

    print("\njoint position:")
    print(robot.data.joint_pos[0])

    print("=================================\n")

    env.close()


if __name__ == "__main__":
    main()