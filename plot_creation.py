import matplotlib.pyplot as plt
from find_foef import x, y


# Построение графика
def plot_creation():
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))  # 3 подграфика в одной строке

    for i in range(3):
        # plt.figure(i + 1)  # Создание нового окна для каждого графика

        axes[i].plot(x[i], y[i], 'o-')
        axes[i].set_title(f'График дисторсии {i}')
        axes[i].set_xlabel('Δ%')
        axes[i].set_ylabel('\u03C3²')
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()
