from marllib import marl

env = marl.make_env(environment_name="aircombat", map_name="MultipleCombat_4v4/NoWeapon/vsBaseline")
# initialize algorithm with appointed hyperparameters
mappo = marl.algos.mappo(hyperparam_source="common")
# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})
# start training
mappo.fit(env, model, stop={"timesteps_total": 1000000}, checkpoint_freq=500, share_policy="group")
