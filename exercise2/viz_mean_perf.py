# training_part1_1.pickle

import pickle
import matplotlib.pyplot as plt

test_time = 5

for part in [1, 2, 3]:
    fig, ax = plt.subplots(figsize=(6, 4))

    for i in range(1, 1 + test_time):
        with open('training_part{}_{}.pickle'.format(part, i), "rb") as f:
            mean_score = pickle.load(f)
            ax.plot(mean_score, label='Trial {}'.format(i))
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlabel('Episode number (x100)', fontsize=10, fontweight='bold')
    ax.set_ylabel(
        'Mean survival time over last x100 episodes', fontsize=10, fontweight='bold'
    )
    if part == 2:
        ax.set_ylim(bottom=0, top=50)
    elif part == 3:
        ax.set_ylim(bottom=0, top=20)
    else:
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig('{}_res.pdf'.format(part), bbox_inches='tight', dpi=300)
