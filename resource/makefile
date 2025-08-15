
all:

test:
	python3 main.py --mode=test --solver=random
	python3 main.py --mode=test --solver=opt
	python3 main.py --mode=test --solver=sac

prepare:
	python3 main.py --scenario=prepare_knapsack --mode=train
	python3 main.py --scenario=prepare_knapsack --mode=test

plot_test:
	python3 main.py --scenario=box_plot --metric=reward
	python3 main.py --scenario=box_plot --metric=objective
	python3 main.py --scenario=box_plot --metric=constraint
	python3 main.py --scenario=box_plot --metric=compute_time

plot_train:
	python3 main.py --scenario=line_plot_training --metric=critic_loss
	python3 main.py --scenario=line_plot_training --metric=actor_loss
	python3 main.py --scenario=line_plot_training --metric=constraint
	python3 main.py --scenario=line_plot_training --metric=objective
	python3 main.py --scenario=line_plot_training --metric=reward
