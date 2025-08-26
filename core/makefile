
test:
	python3 main.py --solver=random
	python3 main.py --solver=sac
	python3 main.py --solver=greedy
	python3 main.py --solver=optimal

train:
	python3 main.py --solver=sac --mode=train # --autotune_entropy

plot_train:
	python3 main.py --mode=train --scenario=plot_line --metric=actor_loss --n_smooth=20 & \
	python3 main.py --mode=train --scenario=plot_line --metric=qf_loss --n_smooth=20 & \
	python3 main.py --mode=train --scenario=plot_line --metric=reward --n_smooth=20 & \
	wait
