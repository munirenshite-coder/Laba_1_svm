import numpy as np

# Пункт 1. Вспомогательные функции

def get_residual_norm(A, x, b):
    #вычисляет норму вектора невязки
    r = A @ x - b
    return np.linalg.norm(r)

def get_error_norm(x_method, x_lib):
    #вычисляет отличие от библиотечного решения
    return np.linalg.norm(x_method - x_lib)

# Пункт 2. Метод Гаусса с выбором ведущего элемента

def solve_gauss(A_input, b_input):
    # создаем копии, чтобы не изменить исходные данные
    A = A_input.astype(float).copy()
    b = b_input.astype(float).copy()
    n = len(b)

    for k in range(n):
        # поиск ведущего элемента в текущем столбце среди необработанных строк
        max_idx = k + np.argmax(np.abs(A[k:n, k]))

        #перестановка строк матрицы A и вектора b
        if max_idx != k:
            A[[k, max_idx]] = A[[max_idx, k]]
            b[[k, max_idx]] = b[[max_idx, k]]

        # проверка на ноль
        if np.abs(A[k, k]) < 1e-12:
            raise ValueError("Матрица вырождена или близка к этому. Деление на ноль.")

        # исключение неизвестных
        for i in range(k + 1, n):
            m_ik = A[i, k] / A[k, k]
            A[i, k:] -= m_ik * A[k, k:]
            b[i] -= m_ik * b[k]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x

# пункт 3. Итерационные методы

def solve_simple_iteration(A, b, x0, eps=1e-6, kmax=1000, verbose=False):
    #метод простой итерации
    n = len(b)
    x_old = x0.astype(float).copy()
    x_new = np.zeros(n)

    if verbose:
        print("\n Метод простой итерации")

    for k in range(1, kmax + 1):
        for i in range(n):
            s = sum(A[i, j] * x_old[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]

        diff = np.linalg.norm(x_new - x_old)

        if verbose:
            print(f"Итерация {k}: x = {np.round(x_new, 4)}, изменение = {diff:.2e}")

        if diff < eps:
            return x_new, k

        x_old = x_new.copy()

    print("Внимание: метод простой итерации не достиг точности за kmax шагов.")
    return x_new, kmax


def solve_seidel(A, b, x0, eps=1e-6, kmax=1000, verbose=False):
    # методом Зейделя
    n = len(b)
    x = x0.astype(float).copy()

    if verbose:
        print("\n Метод Зейделя")

    for k in range(1, kmax + 1):
        x_old = x.copy()

        for i in range(n):
            # в методе Зейделя используем уже обновленные значения вектора x
            s1 = sum(A[i, j] * x[j] for j in range(i))
            s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i, i]

        diff = np.linalg.norm(x - x_old)

        if verbose:
            print(f"Итерация {k}: x = {np.round(x, 4)}, изменение = {diff:.2e}")

        if diff < eps:
            return x, k

    print("Внимание: метод Зейделя не достиг точности за kmax шагов.")
    return x, kmax

# Пункт 4. Вычислительные эксперименты

def run_experiments():
    # Матрица коэффициентов, вектор для варианта 1
    A = np.array([
        [10.35, -1.24,  2.18,  0.75],
        [ 1.12, 11.48, -1.35,  2.06],
        [ 2.31, -1.17, 12.63, -1.42],
        [ 0.84,  2.25, -1.16,  9.74]
    ])
    b = np.array([14.27, 18.64, 20.15, 12.83])

    n = len(b)

    # 4.1 Библиотечное решение
    print("\n")
    print("ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ")
    print("\n")
    print("Определитель матрицы det(A):", np.linalg.det(A))
    print("Число обусловленности cond(A):", np.linalg.cond(A))

    x_lib = np.linalg.solve(A, b)
    print("\nБиблиотечное решение NumPy (x_lib):", x_lib)
    print("Норма невязки:", get_residual_norm(A, x_lib, b))

    # 4.2 Прямой метод (Гаусс)
    print("\n")
    print("ПРЯМОЙ МЕТОД (ГАУСС)")
    x_gauss = solve_gauss(A, b)
    print("Решение методом Гаусса:", x_gauss)
    print("Норма невязки:", get_residual_norm(A, x_gauss, b))
    print("Отличие от NumPy (Δ):", get_error_norm(x_gauss, x_lib))

    # Демонстрация процесса итераций
    print("\n")
    print("Демонстрация пошагового выполнения итераций (на примере Зейделя, eps=1e-4)")
    solve_seidel(A, b, np.zeros(n), eps=1e-4, verbose=True)

    # 4.3 Эксперимент 1: Начальные приближения
    print("\n")
    print("ЭКСПЕРИМЕНТ 1: влияние начального положения (eps=1e-6)")
    x0_variants = {
        "Нулевой вектор": np.zeros(n),
        "Единичный вектор": np.ones(n),
        "Произвольный вектор": np.array([5.0, -5.0, 5.0, -5.0])
    }

    for name, x0 in x0_variants.items():
        print(f"\nНачальное приближение: {name} {x0}")

        x_simp, k_simp = solve_simple_iteration(A, b, x0, eps=1e-6)
        r_simp = get_residual_norm(A, x_simp, b)
        print(f"  Простая итерация: итераций = {k_simp}, невязка = {r_simp:.2e}")

        x_seid, k_seid = solve_seidel(A, b, x0, eps=1e-6)
        r_seid = get_residual_norm(A, x_seid, b)
        print(f"  Метод Зейделя:    итераций = {k_seid}, невязка = {r_seid:.2e}")

    # 4.4 Эксперимент 2: Различная точность eps
    print("\n")
    print("ЭКСПЕРИМЕНТ 2: Влияние требуемой точности (x0 = нулевой вектор)")
    eps_variants = [1e-2, 1e-4, 1e-6]
    x0_zeros = np.zeros(n)

    for eps in eps_variants:
        print(f"\nТребуемая точность eps = {eps}:")
        x_simp, k_simp = solve_simple_iteration(A, b, x0_zeros, eps=eps)
        r_simp = get_residual_norm(A, x_simp, b)
        print(f"  Простая итерация: итераций = {k_simp}, невязка = {r_simp:.2e}")

        x_seid, k_seid = solve_seidel(A, b, x0_zeros, eps=eps)
        r_seid = get_residual_norm(A, x_seid, b)
        print(f"  Метод Зейделя:    итераций = {k_seid}, невязка = {r_seid:.2e}")

    # 4.5 Эксперимент 3: Влияние округления коэффициентов
    print("\n")
    print("ЭКСПЕРИМЕНТ 3: исследование чувствителности к округлению данных ")
    print(f"Число обусловленности исходной матрицы cond(A) = {np.linalg.cond(A):.4f}")

    for decimals in [4, 2]:
        A_rounded = np.round(A, decimals)
        b_rounded = np.round(b, decimals)

        x_rounded = np.linalg.solve(A_rounded, b_rounded)
        diff = np.linalg.norm(x_lib - x_rounded)
        r_rounded = get_residual_norm(A_rounded, x_rounded, b_rounded)
        print(f"Округление до {decimals} знаков:")
        print(f"  Отличие решения от точного Δx = {diff:.2e}")
        print(f"  Норма невязки округленной системы = {r_rounded:.2e}")

if __name__ == "__main__":
    run_experiments()

