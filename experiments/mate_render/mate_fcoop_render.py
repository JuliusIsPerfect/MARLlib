from marllib import marl

env = marl.make_env(environment_name="mate", map_name="MATE-4v2-9-v0", coop_team="target", force_coop=True)
# env = marl.make_env(environment_name="mate", map_name="MATE-4v2-9-v0", coop_team="camera")
# initialize algorithm with appointed hyperparameters
mappo = marl.algos.mappo(hyperparam_source="common")
# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

model_path = ("D:\projs\MARLlib\experiments\mate\exp_results\mappo_mlp_MATE-4v2-9-v0" +
              "\MAPPOTrainer_mate_MATE-4v2-9-v0_2de53_00000_0_2024-01-08_10-32-34\checkpoint_000120\checkpoint-120")
params_path = ("D:\projs\MARLlib\experiments\mate\exp_results\mappo_mlp_MATE-4v2-9-v0" +
               "\MAPPOTrainer_mate_MATE-4v2-9-v0_2de53_00000_0_2024-01-08_10-32-34\params.json")
model_path.replace("\\", "/")
params_path.replace("\\", "/")

# start training
mappo.render(env, model, stop={"timesteps_total": 1000000},
          restore_path={'params_path': params_path,  # experiment configuration
                        'model_path': model_path,  # checkpoint path
                        'render': True},  # render
          local_mode=True, share_policy="group")
