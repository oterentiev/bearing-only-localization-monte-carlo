# Дослідження методом Монте-Карло двовимірної локалізації об'єкта за кутовими вимірюваннями 
# із застосуванням зваженого нелінійного методу найменших квадратів
#
# Основна програма: python kalman_bearing_experiment.py
# Результати програми зберігаються, у цій самій папці де і програма, у вигляді: 
# - CSV файлу
# - два рисунки

# Використання анотації типів у програмі без їх негайного обчислення
from __future__ import annotations

# Підключаємо бібліотеку для роботи з CSV-файлами та таблицями
import csv

# Підключаємо засоби для роботи з файлами та папками
from pathlib import Path

# Підключаємо бібліотеку для математичних обчислень
import numpy as np

# Підключаємо бібліотеку для побудови графіків
import matplotlib.pyplot as plt

# Підключаємо розподіл хі-квадрат для розрахунку довірчої області
from scipy.stats import chi2

# Підключаємо засіб для побудови еліпсів на графіках
from matplotlib.patches import Ellipse

# Задаємо початкові параметри програми
SEED = 20261120

# Кількість експериментів, які будемо виконувати
N_MC = 10_000

# Задаємо координати об'єкта
# одиниця виміру для задачі - метри
TRUE_POSITION = np.array([367.0, 214.0])  
INITIAL_GUESS = np.array([369.0, 213.5]) 

# Визначаємо шлях до папки, в якій знаходиться файл програми.
# У цій папці будуть зберігатися результати обчислень.
OUTPUT_DIR = Path(__file__).resolve().parent

# -----------------------------------------------------------------
# Функція для розрахунку кутів до об'єкта від кожного пункту спостереження
# position — координати об'єкта, тобто X та Y.
# stations_x — координати X пунктів спостереження.
# position[0] — координата X об'єкта.
# position[1] — координата Y об'єкта.
def angle_model(position: np.ndarray, stations_x: np.ndarray) -> np.ndarray:
    # Кут напрямку на об'єкт (пеленг) від кожного пункту спостереження, виражений у радіанах, у діапазоні від 0 до pi (180)
    return np.arctan2(position[1], position[0] - stations_x)

# -----------------------------------------------------------------
# Функція для обчислення матриці Якобі моделі кутових вимірювань за координатами X та Y об'єкта.
def jacobian(position: np.ndarray, stations_x: np.ndarray) -> np.ndarray:
    # Знаходимо різницю між координатою об'єкта та пунктами спостереження
    dx = position[0] - stations_x
    # Обчислюємо знаменник для визначення часткових похідних
    denominator = dx**2 + position[1]**2
    # Формуємо матрицю Якобі за частковими похідними
    return np.column_stack((-position[1] / denominator, dx / denominator))

# -----------------------------------------------------------------
# Функція для визначення найменшої різниці між двома кутами
def wrapped_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Обчислюємо різницю між кутами з урахуванням повного кола (360 градусів)
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi

# -----------------------------------------------------------------
# Функція для визначення координат об'єкта та оцінювання точності результату
def estimate_position(
    observations: np.ndarray,
    stations_x: np.ndarray,
    covariance_measurement: np.ndarray,
    weighted: bool = True,
) -> tuple[np.ndarray, np.ndarray]:

    # Створюємо одиничну матрицю для використання у незваженому методі
    identity = np.eye(len(stations_x))

    # Визначаємо матрицю ваг залежно від обраного способу розрахунку
    weight = np.linalg.inv(covariance_measurement) if weighted else identity

    # Задаємо початкові координати об'єкта для початку розрахунків
    position = INITIAL_GUESS.astype(float).copy()

    # Виконуємо послідовне уточнення координат (не більше 60 ітерацій)
    for _ in range(60):

        # Визначаємо різницю між виміряними та розрахованими кутами
        residual = wrapped_difference(observations, angle_model(position, stations_x))

        # Розраховуємо матрицю Якобі для поточного положення об'єкта
        j_matrix = jacobian(position, stations_x)

        # Формуємо інформаційну матрицю на основі матриці Якобі та ваг
        information = j_matrix.T @ weight @ j_matrix

        # Обчислюємо напрямок і величину поправки для координат
        gradient = j_matrix.T @ weight @ residual
        delta = np.linalg.solve(information, gradient)

        # Розраховуємо поточне значення цільової функції
        old_objective = float(residual @ weight @ residual)

        # Задаємо початковий крок для уточнення координат
        step = 1.0

        # Перевіряємо, чи покращується результат при зміні координат
        while step > 1e-7:
            candidate = position + step * delta
            # Розраховуємо нові кути для запропонованих координат
            candidate_residual = wrapped_difference(
                observations, angle_model(candidate, stations_x)
            )

            # Перевіряємо, чи є нові координати допустимими та чи не збільшується значення цільової функції
            if (np.isfinite(candidate).all()
                    and float(candidate_residual @ weight @ candidate_residual)
                    <= old_objective + 1e-14):
                break

            # Якщо умова не виконується, зменшуємо крок удвічі
            step *= 0.5

        # Завершуємо розрахунок, якщо відповідний крок став занадто малим
        if step <= 1e-7:
            break

        # Оновлюємо координати об'єкта
        position = candidate

        # Перевіряємо, чи достатньо малою стала чергова поправка
        if np.linalg.norm(step * delta) < 1e-9:
            break

    # Повторно розраховуємо матрицю Якобі для остаточних координат
    j_matrix = jacobian(position, stations_x)

    # Формуємо інформаційну матрицю для остаточної оцінки
    information = j_matrix.T @ weight @ j_matrix

    # Обчислюємо обернену інформаційну матрицю
    information_inverse = np.linalg.inv(information)

    # Для зваженого методу визначаємо коваріаційну матрицю оцінки
    if weighted:
        estimator_covariance = information_inverse

    # Для незваженого методу використовуємо іншу формулу оцінювання похибки
    else:
        estimator_covariance = (
            information_inverse
            @ (j_matrix.T @ covariance_measurement @ j_matrix)
            @ information_inverse
        )

    # Повертаємо отримані координати та коваріаційну матрицю
    return position, estimator_covariance

# -----------------------------------------------------------------

# Функція для проведення обчислювального експерименту
# та порівняння результатів різних методів оцінювання координат
def run_configuration(
    stations_x: list[float], sigma_degrees: list[float], rng: np.random.Generator
) -> tuple[dict, dict]:

    # Перетворюємо координати пунктів спостереження у масив
    stations = np.asarray(stations_x, dtype=float)

    # Переводимо похибки вимірювань із градусів у радіани
    sigma_radians = np.deg2rad(np.asarray(sigma_degrees, dtype=float))

    # Формуємо коваріаційну матрицю похибок вимірювань
    measurement_covariance = np.diag(sigma_radians**2)

    # Розраховуємо кути без урахування випадкових похибок
    noiseless_angles = angle_model(TRUE_POSITION, stations)

    # Створюємо списки для збереження результатів обох методів
    residuals: dict[str, list[np.ndarray]] = {"WLS": [], "OLS": []}
    predicted_covariances: dict[str, list[np.ndarray]] = {"WLS": [], "OLS": []}
    estimates: dict[str, list[np.ndarray]] = {"WLS": [], "OLS": []}

    # Повторюємо експеримент задану кількість разів
    for _ in range(N_MC):

        # Додаємо випадкові похибки до кутових вимірювань
        sample = noiseless_angles + rng.normal(0.0, sigma_radians)

        # Порівнюємо зважений та незважений методи оцінювання
        for label, weighted in (("WLS", True), ("OLS", False)):

            # Розраховуємо координати об'єкта та коваріаційну матрицю
            estimate, covariance = estimate_position(
                sample, stations, measurement_covariance, weighted=weighted
            )

            # Зберігаємо отримані координати об'єкта
            estimates[label].append(estimate)

            # Обчислюємо похибки визначення координат
            residuals[label].append(estimate - TRUE_POSITION)

            # Зберігаємо розраховані коваріаційні матриці
            predicted_covariances[label].append(covariance)

    # Створюємо словники для збереження підсумкових результатів
    summary: dict[str, dict] = {}
    samples: dict[str, dict] = {}

    # Окремо обробляємо результати кожного методу
    for label in ("WLS", "OLS"):

        # Перетворюємо отримані результати у масиви
        errors = np.asarray(residuals[label])
        estimates_array = np.asarray(estimates[label])
        covariance_array = np.asarray(predicted_covariances[label])

        # Розраховуємо фактичну коваріаційну матрицю похибок
        empirical_covariance = np.cov(errors.T, ddof=1)

        # Знаходимо середню прогнозовану коваріаційну матрицю
        mean_predicted_covariance = covariance_array.mean(axis=0)

        # Перевіряємо відповідність похибок розрахованим коваріаціям
        nees = np.asarray([
            error @ np.linalg.solve(cov, error)
            for error, cov in zip(errors, covariance_array)
        ])

        # Розраховуємо основні показники точності та невизначеності
        summary[label] = {
            "n": N_MC,
            "bias_x_m": float(errors[:, 0].mean()),
            "bias_y_m": float(errors[:, 1].mean()),
            "rmse_x_m": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
            "rmse_y_m": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
            "rmse_position_m": float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
            "empirical_sd_x_m": float(np.sqrt(empirical_covariance[0, 0])),
            "empirical_sd_y_m": float(np.sqrt(empirical_covariance[1, 1])),
            "predicted_sd_x_m": float(np.sqrt(mean_predicted_covariance[0, 0])),
            "predicted_sd_y_m": float(np.sqrt(mean_predicted_covariance[1, 1])),

            # Визначаємо частку результатів, що потрапили до 95% довірчої області
            "coverage_95": float(np.mean(nees <= chi2.ppf(0.95, df=2))),
        }

        # Зберігаємо результати для подальшої побудови графіків
        samples[label] = {
            "errors": errors,
            "estimates": estimates_array,
            "mean_covariance": mean_predicted_covariance,
        }

    # Повертаємо підсумкові показники та результати моделювання
    return summary, {"stations": stations, "sigma_degrees": sigma_degrees,
                     "noiseless_angles_degrees": np.rad2deg(noiseless_angles),
                     **samples}


# Функція для побудови еліпса, який характеризує невизначеність оцінки координат
def add_covariance_ellipse(ax, covariance: np.ndarray, confidence: float = 0.95) -> None:

    # Знаходимо власні значення та власні вектори коваріаційної матриці
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Розташовуємо власні значення у порядку спадання
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Визначаємо коефіцієнт для заданого рівня довіри
    scale = chi2.ppf(confidence, df=2)

    # Розраховуємо довжину великої та малої осей еліпса
    width, height = 2.0 * np.sqrt(eigenvalues * scale)

    # Визначаємо кут нахилу еліпса
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Додаємо еліпс на графік
    ax.add_patch(Ellipse((0.0, 0.0), width, height, angle=angle,
                         fill=False, linewidth=2, label="Теоретична 95% область"))


# Функція для побудови схеми розташування об'єкта та пунктів спостереження
def make_geometry_figure(path: Path) -> None:

    # Створюємо область для побудови графіка
    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    # Задаємо координати об'єкта
    target = TRUE_POSITION

    # Вказуємо координати чотирьох пунктів спостереження
    stations = np.array([0.0, 152.5, 305.0, 457.5])

    # Послідовно відображаємо кожний пункт спостереження
    for idx, station in enumerate(stations):

        # Позначаємо пункт спостереження на графіку
        ax.scatter(station, 0, marker="s", s=55,
                   label="Пункти спостереження" if idx == 0 else None)

        # Додаємо підписи з координатами пунктів спостереження
        ax.text(station, -12, f"B{idx+1} ({station:g}; 0)",
                ha="center", va="top", fontsize=8)

        # Для четвертого пункту використовуємо штрихову лінію
        style = "--" if idx == 3 else "-"

        # Будуємо лінії напрямку від пунктів до об'єкта
        ax.plot([station, target[0]], [0, target[1]], linestyle=style,
                linewidth=1.5, alpha=0.85)

    # Позначаємо справжнє положення об'єкта
    ax.scatter(*target, marker="*", s=180,
               label="Об'єкт A (367; 214)", zorder=5)

    # Додаємо підпис об'єкта
    ax.annotate("A (367; 214)", target, xytext=(8, 8),
                textcoords="offset points")

    # Підписуємо горизонтальну та вертикальну осі
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")

    # Додаємо назву графіка
    ax.set_title("Геометрія кутових вимірювань")

    # Встановлюємо межі відображення графіка
    ax.set_xlim(-25, 500)
    ax.set_ylim(-25, 245)

    # Додаємо координатну сітку та легенду
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    # Налаштовуємо розташування елементів графіка
    fig.tight_layout()

    # Зберігаємо графік у файл
    fig.savefig(path, dpi=220)

    # Закриваємо графік після збереження
    plt.close(fig)


# Функція для побудови графіків розсіювання похибок координат
# для трьох та чотирьох пунктів спостереження
def make_uncertainty_figures(samples: dict, path3: Path, path4: Path) -> None:

    # Послідовно обробляємо результати для двох конфігурацій
    for key, path, title in (
        ("three", path3, "Три пункти спостереження"),
        ("four", path4, "Чотири пункти спостереження"),
    ):

        # Отримуємо похибки оцінювання координат методом WLS
        errors = samples[key]["WLS"]["errors"]

        # Отримуємо середню коваріаційну матрицю
        covariance = samples[key]["WLS"]["mean_covariance"]

        # Створюємо область для побудови графіка
        fig, ax = plt.subplots(figsize=(5.8, 4.7))

        # Відображаємо частину отриманих результатів моделювання
        ax.scatter(errors[::5, 0], errors[::5, 1], s=5, alpha=0.18,
                   label="Оцінки Монте-Карло")

        # Позначаємо істинне положення об'єкта в системі координат похибок
        ax.scatter(0, 0, marker="+", s=100, linewidths=2,
                   label="Істинне положення")

        # Додаємо 95-відсотковий еліпс невизначеності
        add_covariance_ellipse(ax, covariance)

        # Підписуємо осі графіка
        ax.set_xlabel("Похибка координати x, м")
        ax.set_ylabel("Похибка координати y, м")

        # Додаємо назву графіка
        ax.set_title(title)

        # Додаємо сітку та легенду
        ax.grid(True, alpha=0.25)
        ax.axis("equal")
        ax.legend(fontsize=8)

        # Налаштовуємо розташування елементів графіка
        fig.tight_layout()

        # Зберігаємо графік у відповідний файл
        fig.savefig(path, dpi=220)

        # Закриваємо графік після збереження
        plt.close(fig)


# Функція для збереження результатів експерименту у CSV-файл
def save_summary_csv(path: Path, summaries: dict) -> None:

    # Визначаємо назви стовпців майбутньої таблиці
    fieldnames = ["configuration", "method", "n", "bias_x_m", "bias_y_m",
                  "rmse_x_m", "rmse_y_m", "rmse_position_m",
                  "empirical_sd_x_m", "empirical_sd_y_m",
                  "predicted_sd_x_m", "predicted_sd_y_m", "coverage_95"]

    # Відкриваємо файл для запису результатів
    with path.open("w", newline="", encoding="utf-8-sig") as f:

        # Створюємо засіб для запису табличних даних
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Записуємо назви стовпців
        writer.writeheader()

        # Послідовно записуємо результати для кожної конфігурації
        for configuration, methods in summaries.items():

            # Записуємо результати окремо для кожного методу
            for method, values in methods.items():
                writer.writerow({"configuration": configuration, "method": method, **values})


# Функція для перевірки роботи алгоритму на конкретному чисельному прикладі
def deterministic_example() -> tuple[np.ndarray, np.ndarray]:

    # Задаємо координати трьох пунктів спостереження
    stations = np.array([0.0, 152.5, 305.0])

    # Вказуємо стандартні відхилення кутових вимірювань
    sigma = np.deg2rad(np.array([0.10, 0.10, 0.20]))

    # Формуємо коваріаційну матрицю вимірювань
    covariance = np.diag(sigma**2)

    # Задаємо відомі кути спостереження з вихідного прикладу
    observations = np.deg2rad(np.array([30.1, 45.0, 73.6]))

    # Розраховуємо координати об'єкта за допомогою зваженого методу
    return estimate_position(observations, stations, covariance, weighted=True)


# Головна функція програми, яка запускає обчислювальний експеримент
def main() -> None:

    # Створюємо генератор випадкових чисел із заданим початковим значенням
    rng = np.random.default_rng(SEED)

    # Проводимо експеримент для трьох пунктів спостереження
    summary3, sample3 = run_configuration(
        [0.0, 152.5, 305.0], [0.10, 0.10, 0.20], rng
    )

    # Проводимо експеримент для чотирьох пунктів спостереження
    summary4, sample4 = run_configuration(
        [0.0, 152.5, 305.0, 457.5], [0.10, 0.10, 0.20, 0.10], rng
    )

    # Об'єднуємо результати обох експериментів
    summaries = {"3 stations": summary3, "4 stations": summary4}

    # Зберігаємо підсумкові результати у CSV-файл
    save_summary_csv(OUTPUT_DIR / "monte_carlo_summary.csv", summaries)

    # Будуємо та зберігаємо схему розташування пунктів спостереження
    make_geometry_figure(OUTPUT_DIR / "bearing_geometry.png")

    # Будуємо графіки розсіювання похибок для обох конфігурацій
    make_uncertainty_figures(
        {"three": sample3, "four": sample4},
        OUTPUT_DIR / "uncertainty_3_stations.png",
        OUTPUT_DIR / "uncertainty_4_stations.png",
    )

    # Виконуємо контрольний розрахунок за відомими кутовими вимірюваннями
    example_position, example_covariance = deterministic_example()

    # Виводимо початкові параметри експерименту
    print(f"Seed: {SEED}; Monte Carlo trials per configuration: {N_MC}")

    # Виводимо розраховані кути для трьох та чотирьох пунктів спостереження
    print("Noiseless angles (3 stations), deg:", sample3["noiseless_angles_degrees"])
    print("Noiseless angles (4 stations), deg:", sample4["noiseless_angles_degrees"])

    # Виводимо підсумкові показники для кожної конфігурації та методу
    for configuration, methods in summaries.items():
        for method, metrics in methods.items():
            print(configuration, method, metrics)

    # Виводимо результати контрольного розрахунку
    print("Reference example estimate, metres:", example_position)
    print("Reference example covariance, m^2:\n", example_covariance)

    # Вказуємо папку, у якій збережено результати роботи програми
    print("Outputs written to:", OUTPUT_DIR)




if __name__ == "__main__":
    main()
