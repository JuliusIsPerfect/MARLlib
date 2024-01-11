from marllib import marl

env = marl.make_env(environment_name="aircombat", map_name="MultipleCombat_4v4/NoWeapon/vsBaseline")
# initialize algorithm with appointed hyperparameters
mappo = marl.algos.mappo(hyperparam_source="common")
# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})
# start training
# mappo.fit(env, model, stop={"timesteps_total": 1000000}, checkpoint_freq=10, share_policy="group")
mappo.render(env, model, stop={"timesteps_total": 1000000}, checkpoint_freq=10,
             restore_path={'params_path': "D:/projs/MARLlib/experiments/exp_results/mappo_mlp_MultipleCombat_4v4/NoWeapon/vsBaseline/MAPPOTrainer_aircombat_MultipleCombat_4v4_NoWeapon_vsBaseline_370a4_00000_0_2024-01-04_09-33-08/params.json",
                 'model_path': "D:/projs/MARLlib/experiments/exp_results/mappo_mlp_MultipleCombat_4v4/NoWeapon/vsBaseline/MAPPOTrainer_aircombat_MultipleCombat_4v4_NoWeapon_vsBaseline_370a4_00000_0_2024-01-04_09-33-08/checkpoint_000020/checkpoint-20",
                           'render': True}, share_policy="group")
