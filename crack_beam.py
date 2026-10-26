# -*- coding: utf-8 -*-
"""
Расчет малоармированной изгибаемой статически неопределимой балки
с единичной трещиной.

Версия 3.0 — новый интерфейс в стиле Apple (customtkinter): боковая панель,
карточки, живая схема, эпюры, кольца нагруженности, светлая и тёмная темы.
Модель:
    слева жесткая заделка, справа шарнирно-подвижная опора;
    один пролет;
    равномерная нагрузка = собственный вес + заданная пользователем.

Единицы внутри расчета:
    длины - м
    площади - м²
    объем - м³
    силы - Н
    моменты - Н·м
    напряжения/модули - Па

Вывод:
    все основные промежуточные результаты, подобранный диаметр,
    As, Mcrc, параметры Пересыпкина, итерации.

ВАЖНО:
    Таблица графиков Пересыпкина ниже является ручной
    оцифровкой кривых с предоставленного изображения; между узловыми
    точками используется линейная интерполяция, без полиномиальной
    аппроксимации.
"""

import math
import json
import os
import re
import textwrap
import time
import tkinter as tk
from tkinter import messagebox, filedialog
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFilter
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ----------------------------------------------------------------------
# Материалы
# ----------------------------------------------------------------------

@dataclass
class Concrete:
    name: str
    Eb: float      # Pa
    Rb: float      # Pa
    Rbt: float     # Pa
    Rb_ser: float  # Pa
    psi_b: float


@dataclass
class Steel:
    name: str
    Es: float      # Pa
    Rs: float      # Pa
    Rsc: float     # Pa
    Ea: float      # Pa, по условию задания
    psi_s: float


# Значения для тяжелого бетона, приведенные к SI.
# Rb - расчетное сопротивление сжатию;
# Rb_ser - сопротивление для II группы.
CONCRETES = {
    "B15": Concrete("B15", 24.0e9, 8.5e6, 0.75e6, 11.0e6, 0.90),
    "B20": Concrete("B20", 27.0e9, 11.5e6, 0.90e6, 15.0e6, 0.90),
    "B25": Concrete("B25", 30.0e9, 14.5e6, 1.05e6, 18.5e6, 0.90),
    "B30": Concrete("B30", 32.5e9, 17.0e6, 1.15e6, 22.0e6, 0.90),
    "B35": Concrete("B35", 34.5e9, 19.5e6, 1.30e6, 25.5e6, 0.90),
    "B40": Concrete("B40", 36.0e9, 22.0e6, 1.40e6, 29.0e6, 0.90),
    "B45": Concrete("B45", 37.0e9, 25.0e6, 1.50e6, 32.0e6, 0.90),
    "B50": Concrete("B50", 39.0e9, 27.5e6, 1.60e6, 36.0e6, 0.90),
    "B55": Concrete("B55", 39.5e9, 30.0e6, 1.70e6, 39.5e6, 0.90),
    "B60": Concrete("B60", 40.0e9, 33.0e6, 1.80e6, 43.0e6, 0.90),
}

# Для первой версии оставлены основные ненапрягаемые классы,
# применимые к рабочей арматуре.
STEELS = {
    # Ea — предел текучести/условный предел текучести, а не модуль Es.
    "A240": Steel("A240", 200.0e9, 210.0e6, 210.0e6, 210.0e6, 1.0),
    "A400": Steel("A400", 200.0e9, 340.0e6, 340.0e6, 340.0e6, 1.0),
    "A500": Steel("A500", 200.0e9, 435.0e6, 435.0e6, 435.0e6, 1.0),
    "A600": Steel("A600", 200.0e9, 520.0e6, 470.0e6, 520.0e6, 1.0),
    "B500": Steel("B500", 200.0e9, 415.0e6, 415.0e6, 415.0e6, 1.0),
}

# Сортамент арматуры по предоставленной пользователем таблице.
# BAR_AREAS_MM2 — площадь ОДНОГО стержня в мм².
# Внутри расчёта площади переводятся в м².
BAR_AREAS_MM2 = {
    3: 7.1,
    4: 12.6,
    5: 19.6,
    6: 28.3,
    7: 38.5,
    8: 50.3,
    9: 63.6,
    10: 78.5,
    12: 113.1,
    14: 153.9,
    16: 201.1,
    18: 254.5,
    20: 314.2,
    22: 380.1,
    25: 490.9,
    28: 615.8,
    32: 804.2,
    36: 1018.0,
    40: 1256.0,
}
DIAMETERS_MM = sorted(BAR_AREAS_MM2)

# Настройки итерационного процесса. Формулы и критерий 5% сохранены.
MAX_ITERATIONS = 20
MOMENT_TOLERANCE_PERCENT = 5.0


def _select_reinforcement(As_required, n):
    """Минимальный диаметр из табличного сортамента с As >= As_required."""
    if not math.isfinite(As_required) or As_required <= 0:
        raise ValueError(
            f"Требуемая площадь арматуры должна быть положительной и конечной; "
            f"получено As0={As_required!r} м²."
        )
    for d_mm in DIAMETERS_MM:
        As_provided = n * BAR_AREAS_MM2[d_mm] / 1_000_000.0
        if As_provided >= As_required:
            return d_mm / 1000.0, As_provided
    max_area = n * BAR_AREAS_MM2[DIAMETERS_MM[-1]]
    raise ValueError(
        f"Не удалось подобрать арматуру: требуемая As0={As_required * 1e6:.3f} мм², "
        f"а максимальная площадь {n} стержней Ø{DIAMETERS_MM[-1]} мм "
        f"по таблице составляет {max_area:.3f} мм²."
    )


# ----------------------------------------------------------------------
# Графики Пересыпкина
# ----------------------------------------------------------------------
# Узловые точки (S/h0, L) для кривых α0m.
# Значения получены ручной оцифровкой предоставленного графика.
# Линейная интерполяция используется только между точками.
#
# Ключи: alpha0m.
PERE_SYPN_L = {
    0.020: [
        (0.50, 6.80), (0.55, 6.65), (0.60, 6.30), (0.65, 6.0),
        (0.70, 5.35), (0.75, 4.55), (0.80, 3.55), (0.85, 2.05),
        (0.88, 0.75), (0.89, 0.5),
    ],
    0.030: [
        (0.50, 6.20), (0.55, 6.0), (0.60, 5.7), (0.65, 5.2),
        (0.70, 4.55), (0.75, 3.6), (0.80, 2.6), (0.85, 2.0),
        (0.88, 0.3), (0.89, 0),
    ],
    0.038: [
        (0.50, 5.70), (0.55, 5.5), (0.60, 5.2), (0.65, 4.6),
        (0.70, 3.8), (0.75, 2.75), (0.80, 1.85), (0.85, 0.40),
        (0.88, 0.0), (0.89, 0.0),
    ],
    0.013: [
        (0.50, 7.70), (0.55, 7.67), (0.60, 7.6), (0.65, 7.4),
        (0.70, 7.0), (0.75, 6.2), (0.80, 4.8), (0.85, 2.6),
        (0.88, 1.2), (0.89, 0.8),
            ],

}
# Mb/Mcrc = K. На исходном графике K нанесен ниже нулевой линии.
PERE_SYPN_K = {
    0.030: [(0.50, 0.2), (0.60, 0.155), (0.70, 0.105),
            (0.80, 0.045), (0.85, 0), (0.88, 0)],
    0.020: [(0.50, 0.17), (0.60, 0.14), (0.70, 0.09),
            (0.80, 0.03), (0.85, 0.032), (0.88, 0.010)],
    0.013: [(0.50, 0.16), (0.60, 0.12), (0.70, 0.065),
            (0.80, 0.0), (0.85, 0.0), (0.88, 0.0)],
    0.038: [(0.50, 0.21), (0.60, 0.175), (0.70, 0.115),
            (0.80, 0.07), (0.85, 0.02), (0.88, 0.0)],
}

# ξ1(S/h0) по правому графику.
PERE_SYPN_XI = {
    0.020: [(0.50, 0.33), (0.60, 0.28), (0.70, 0.22),
            (0.80, 0.16), (0.90, 0.075)],
    0.030: [(0.50, 0.33), (0.60, 0.29), (0.70, 0.225),
            (0.8, 0.16),(0.9, 0.08)],
    0.038: [(0.50, 0.36), (0.60, 0.31), (0.70, 0.24),
            (0.80, 0.18), (0.90, 0.095)],
    0.013: [(0.50, 0.22), (0.60, 0.16), (0.70, 0.09),
            (0.80, 0.01), (0.90, 0.0)],
}


def linear_interp(points, x):
    points = sorted(points)
    if x < points[0][0] or x > points[-1][0]:
        raise ValueError(
            f"Не удалось интерполировать график Пересыпкина: параметр по оси X равен {x:.4f}, "
            f"а оцифрованный диапазон этой кривой — от {points[0][0]:.3f} до {points[-1][0]:.3f}. "
            "Измените исходные данные или дополните оцифровку по оригинальному графику; экстраполяция не выполняется."
        )
    for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
        if x1 <= x <= x2:
            if abs(x2 - x1) < 1e-14:
                return y1
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    return points[-1][1]


def graph_value(table, alpha, x):
    alphas = sorted(table)
    if alpha < alphas[0] or alpha > alphas[-1]:
        raise ValueError(
            f"Параметр приведённой арматуры α0m={alpha:.5f} не попадает в диапазон "
            f"кривых Пересыпкина ({alphas[0]:.3f}–{alphas[-1]:.3f}). "
            "По имеющимся кривым нельзя надёжно определить параметры трещины. "
            "Проверьте b, h, защитный слой, количество/диаметр арматуры и классы материалов."
        )

    # Сначала ищем совпадение с табличным узлом с допуском для float.
    # Это предотвращает ненужную интерполяцию, например для 0.0130000000001.
    exact_alpha = next((a for a in alphas if math.isclose(alpha, a, rel_tol=0.0, abs_tol=1e-9)), None)
    if exact_alpha is not None:
        return linear_interp(table[exact_alpha], x)

    # Интерполяция между соседними кривыми по α0m.
    for a1, a2 in zip(alphas[:-1], alphas[1:]):
        if a1 <= alpha <= a2:
            y1 = linear_interp(table[a1], x)
            y2 = linear_interp(table[a2], x)
            return y1 + (y2 - y1) * (alpha - a1) / (a2 - a1)

    raise ValueError("Не удалось определить значение по графику.")


# ----------------------------------------------------------------------
# Вспомогательная математика
# ----------------------------------------------------------------------

def integrate_simpson(func, a, b, n=2000):
    if n % 2:
        n += 1
    h = (b - a) / n
    s = func(a) + func(b)
    for i in range(1, n):
        x = a + i * h
        s += (4 if i % 2 else 2) * func(x)
    return s * h / 3.0


def solve_2x2(a11, a12, a21, a22, b1, b2):
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-30:
        raise ValueError("Не удалось решить каноническую систему: её определитель близок к нулю. Это означает, что коэффициенты системы вырождены или имеют некорректные значения. Проверьте эпюры единичных воздействий, жёсткость и положение трещины.")
    x1 = (b1 * a22 - a12 * b2) / det
    x2 = (a11 * b2 - b1 * a21) / det
    return x1, x2


# ----------------------------------------------------------------------
# Статически неопределимая балка
# ----------------------------------------------------------------------
# Один пролет, слева заделка, справа вертикальная подвижная опора.
# Используется эквивалентная силовая форма решения; она дает те же
# M(x), Q(x), что и метод перемещений для данной линейно-упругой схемы.
#
# Для UDL:
# R_B = 3qL/8
# R_A = 5qL/8
# M_A = -qL²/8
#
# q > 0 направлена вниз.
def beam_actions(L, q, x):
    Rb = 3.0 * q * L / 8.0
    Ra = q * L - Rb
    Ma = -q * L * L / 8.0
    M = Ma + Ra * x - q * x * x / 2.0
    Q = Ra - q * x
    N = 0.0
    return M, Q, N, Ra, Rb, Ma


def _unit_epure_coefficients(L, c):
    """Аналитические коэффициенты эпюр для балки: заделка слева, опора справа.

    Зафиксированная система знаков (допущение модели):
    - q направлена вниз;
    - положительный внешний единичный момент m=1 приложен против часовой
      стрелки на правой грани левой части разреза; в принятой эпюре это
      даёт скачок M_m(c+)-M_m(c-)=-1;
    - единичная поперечная сила P=1 направлена вниз; эпюра M_P непрерывна,
      а поперечная сила имеет скачок, соответствующий P=1.

    Реакция правой опоры находится из условия v(L)=0 для основной
    консольной системы. Эпюры на участках [0,c] и [c,L] записаны как
    полиномы по глобальной координате x; коэффициенты идут от свободного
    члена к старшим степеням.

    Для момента: [0,c] M_m=M0m+RA_m*x; [c,L] M_m=M0m-1+RA_m*x.
    Для силы:   [0,c] M_P=M0P+RA_P*x; [c,L] M_P=M0P+c+(RA_P-1)*x.

    Это выбранная и явно зафиксированная конвенция знаков, поскольку в
    текстовом фрагменте методики направления единичных воздействий не
    показаны. Каноническая система решается согласованно с этой конвенцией.
    """
    if not (0.0 < c < L):
        raise ValueError("Трещина должна находиться строго внутри пролета (0 < X < L).")

    # Единичный момент в сечении c. Положительный внешний момент принят
    # против часовой стрелки; знак задает непрерывную систему эпюр.
    RB_m = -3.0 * c * (2.0 * L - c) / (2.0 * L ** 3)
    RA_m = -RB_m
    M0_m = 1.0 + RB_m * L
    M_m_left = np.array([M0_m, RA_m], dtype=float)
    M_m_right = np.array([M0_m - 1.0, RA_m], dtype=float)

    # Единичная вертикальная сила вниз в сечении c.
    RB_p = c ** 2 * (3.0 * L - c) / (2.0 * L ** 3)
    RA_p = 1.0 - RB_p
    M0_p = -c + RB_p * L
    M_p_left = np.array([M0_p, RA_p], dtype=float)
    M_p_right = np.array([M0_p + c, RA_p - 1.0], dtype=float)
    return M_m_left, M_m_right, M_p_left, M_p_right


def _integrate_product_polynomials(a_coeff, b_coeff, x0, x1):
    """Точное аналитическое интегрирование произведения двух полиномов."""
    product = np.polynomial.polynomial.polymul(a_coeff, b_coeff)
    primitive = np.polynomial.polynomial.polyint(product)
    return float(np.polynomial.polynomial.polyval(x1, primitive)
                 - np.polynomial.polynomial.polyval(x0, primitive))


def canonical_coefficients(L, c, E, I, q):
    """Аналитические коэффициенты канонической системы для заданной схемы.

    Балка: заделка x=0, шарнирно-подвижная опора x=L, равномерная нагрузка q.
    Эпюры единичных момента и силы получены из условия v(L)=0 для основной
    консольной системы и используют систему знаков, описанную в
    _unit_epure_coefficients. Все интегралы Мора вычисляются аналитически
    по участкам слева и справа от трещины, без численного интегрирования.
    При использовании другой ориентации единичных воздействий необходимо
    изменить соответствующие знаки эпюр и согласованно трактовать M и Q.
    """
    if not (0.0 < c < L):
        raise ValueError("Трещина должна находиться строго внутри пролета (0 < X < L).")
    EI = E * I
    Mm_l, Mm_r, Mp_l, Mp_r = _unit_epure_coefficients(L, c)
    # M от фактической равномерной нагрузки: M(x)=Ma+Ra*x-q*x²/2.
    _, _, _, Ra, _, Ma = beam_actions(L, q, 0.0)
    Mload = np.array([Ma, Ra, -0.5 * q], dtype=float)

    # Суммирование точных интегралов по участкам до и после трещины.
    dMM = (_integrate_product_polynomials(Mm_l, Mm_l, 0.0, c)
           + _integrate_product_polynomials(Mm_r, Mm_r, c, L)) / EI
    dMQ = (_integrate_product_polynomials(Mm_l, Mp_l, 0.0, c)
           + _integrate_product_polynomials(Mm_r, Mp_r, c, L)) / EI
    dQQ = (_integrate_product_polynomials(Mp_l, Mp_l, 0.0, c)
           + _integrate_product_polynomials(Mp_r, Mp_r, c, L)) / EI
    dMP = (_integrate_product_polynomials(Mm_l, Mload, 0.0, c)
           + _integrate_product_polynomials(Mm_r, Mload, c, L)) / EI
    dQP = (_integrate_product_polynomials(Mp_l, Mload, 0.0, c)
           + _integrate_product_polynomials(Mp_r, Mload, c, L)) / EI
    return dMM, dMQ, dQQ, dMP, dQP


# ----------------------------------------------------------------------
# Основной расчет
# ----------------------------------------------------------------------

def _inverse_L(table, target_L):
    """Инверсия графика L(S/h0) без экстраполяции."""
    pts = sorted(table, key=lambda z: z[0])
    values = [v for _, v in pts]
    Lmin, Lmax = min(values), max(values)
    if target_L < Lmin - 1e-12 or target_L > Lmax + 1e-12:
        return None

    # Точное попадание в узел.
    for s, lv in pts:
        if abs(lv - target_L) < 1e-12:
            return s

    for (s1, L1), (s2, L2) in zip(pts[:-1], pts[1:]):
        if (L1 - target_L) * (L2 - target_L) <= 0:
            if abs(L2 - L1) < 1e-14:
                return (s1 + s2) / 2.0
            return s1 + (s2 - s1) * (target_L - L1) / (L2 - L1)
    return None


def _graph_parameters(alpha0m, Lp):
    """Получить S/h0, K и ξ1 по исходным таблицам графика.

    Используется только линейная интерполяция между оцифрованными
    кривыми и точками. Экстраполяция запрещена.
    """
    alphas = sorted(PERE_SYPN_L)
    if not (alphas[0] <= alpha0m <= alphas[-1]):
        raise ValueError(
            f"α0m={alpha0m:.6f} выходит за диапазон графика "
            f"Пересыпкина [{alphas[0]:.3f}; {alphas[-1]:.3f}]."
        )

    # Если alpha0m практически совпадает с табличным узлом, используем
    # именно эту кривую, не полагаясь на точное сравнение float.
    exact_alpha = next((a for a in alphas if math.isclose(alpha0m, a, rel_tol=0.0, abs_tol=1e-9)), None)

    # Находим S/h0 на одной кривой или интерполяцией между соседними L(S/h0).
    if exact_alpha is not None:
        s_h0 = _inverse_L(PERE_SYPN_L[exact_alpha], Lp)
    else:
        s_h0 = None
        for a1, a2 in zip(alphas[:-1], alphas[1:]):
            if a1 <= alpha0m <= a2:
                s1 = _inverse_L(PERE_SYPN_L[a1], Lp)
                s2 = _inverse_L(PERE_SYPN_L[a2], Lp)
                if s1 is None or s2 is None:
                    return None, None, None
                s_h0 = s1 + (s2 - s1) * (alpha0m - a1) / (a2 - a1)
                break

    if s_h0 is None:
        return None, None, None

    try:
        K = graph_value(PERE_SYPN_K, alpha0m, s_h0)
        xi1 = graph_value(PERE_SYPN_XI, alpha0m, s_h0)
    except ValueError as exc:
        raise ValueError(
            f"Для α0m={alpha0m:.6f} и S/h0={s_h0:.6f} "
            f"невозможно снять K или ξ1 с графика без экстраполяции: {exc}"
        ) from exc

    return s_h0, K, xi1


def _section_state(M0, *, b, h, a, h0, n, chosen_d, As, alpha0,
                   concrete, steel, N0):
    """Все величины, которые должны пересчитываться при новом M0."""
    xi = M0 / (concrete.Rb * b * h0 ** 2)

    # As0 = ξ0 * b * h0 * (Rb/Ea); b — фактическая ширина сечения.
    # ξ0 = M0/(Rb*b*h0*h0).
    As0 = xi * b * h0 * (concrete.Rb / steel.Ea)

    alpha0m = As / (b * h0) * alpha0

    yt = (
        0.5 * b * h ** 2
        + As * alpha0 * a
    ) / (b * h + As)

    d = chosen_d
    Wpl = 1.3 * (
        (
            b * h ** 3 / 12.0
            + b * h * (h / 2.0 - yt) ** 2
            + n * alpha0 * (
                math.pi * d ** 4 / 64.0
                + math.pi * d ** 2 / 4.0 * (yt - a) ** 2
            )
        ) / yt
    )

    # По согласованию с пользователем для момента образования трещин
    # используется сопротивление бетона осевому растяжению Rbt.
    Mcrc = concrete.Rbt * Wpl + N0 * (
        0.76923 * Wpl / (b * h + As)
    )

    # Lp безразмерен:
    # [1.7] = MPa·m^0.5 = N/m^1.5.
    Lp = 1.7e6 * b * h0 ** 1.5 / M0

    return xi, As0, alpha0m, yt, Wpl, Mcrc, Lp


class IterationCalculationError(ValueError):
    """Ошибка расчёта с сохранённым журналом уже выполненных шагов."""
    def __init__(self, message, iterations=None):
        super().__init__(message)
        self.iterations = list(iterations or [])


def calculate(data):
    """Выполняет расчет; исходные формулы и инженерная методика сохранены."""
    required_keys = ("b_mm", "h_mm", "a_mm", "L_m", "q_kN_m", "X_mm", "n", "concrete", "steel")
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError("Не заданы обязательные исходные данные: " + ", ".join(missing))

    numeric_keys = ("b_mm", "h_mm", "a_mm", "L_m", "q_kN_m", "X_mm", "n")
    for key in numeric_keys:
        try:
            value = float(data[key])
        except (TypeError, ValueError):
            raise ValueError(f"Параметр {key} должен быть числом; получено {data[key]!r}.") from None
        if not math.isfinite(value):
            raise ValueError(f"Параметр {key} должен быть конечным числом; получено {data[key]!r}.")

    n_value = float(data["n"])
    if not n_value.is_integer():
        raise ValueError(f"Количество стержней n должно быть целым числом; получено {n_value:g}.")

    try:
        concrete = CONCRETES[data["concrete"]]
    except KeyError:
        raise ValueError(f"Неизвестный класс бетона: {data['concrete']!r}. Допустимые значения: {', '.join(CONCRETES)}.") from None
    try:
        steel = STEELS[data["steel"]]
    except KeyError:
        raise ValueError(f"Неизвестный класс арматуры: {data['steel']!r}. Допустимые значения: {', '.join(STEELS)}.") from None

    b = float(data["b_mm"]) / 1000.0
    h = float(data["h_mm"]) / 1000.0
    a = float(data["a_mm"]) / 1000.0
    L = float(data["L_m"])
    q_user = float(data["q_kN_m"]) * 1000.0
    X = float(data["X_mm"]) / 1000.0
    n = int(n_value)

    if b <= 0 or h <= 0 or L <= 0:
        raise ValueError("Некорректная геометрия: ширина b, высота h и пролёт L должны быть больше нуля. Проверьте введённые значения и единицы измерения (b и h — мм, L — м).")
    if a <= 0 or a >= h:
        raise ValueError("Некорректный защитный слой: a должен быть больше 0 и меньше высоты h. Введите a в миллиметрах и проверьте, что он не равен высоте сечения.")
    if not 0 < X < L:
        raise ValueError(f"Трещина задана вне допустимой области: X={X*1000:.1f} мм, L={L*1000:.1f} мм. Укажите X строго между левой заделкой (0 мм) и правой опорой ({L*1000:.1f} мм).")
    if n <= 0:
        raise ValueError("Количество продольных стержней n должно быть целым числом не меньше 1. Проверьте поле количества стержней.")
    if float(data["q_kN_m"]) < 0:
        raise ValueError("Заданная нагрузка q отрицательна. В этой версии можно вводить только нагрузку, направленную вниз, равную нулю или положительную. Если требуется учитывать подъёмную нагрузку, расчётную схему нужно расширить.")
    if a >= h / 2:
        raise ValueError(f"Защитный слой a={a*1000:.1f} мм слишком велик относительно высоты h={h*1000:.1f} мм: сейчас a составляет не менее половины высоты сечения. Уменьшите a или проверьте введённую высоту h.")

    A = b * h
    I = b * h ** 3 / 12.0
    W = b * h ** 2 / 6.0
    h0 = h - a

    q_self = 25_000.0 * A
    q_total = q_self + q_user

    # M0, Q0, N0 в сечении трещины от внешней нагрузки.
    M0, Q0, N0, Ra, Rb, Ma = beam_actions(L, q_total, X)
    if M0 <= 0:
        raise ValueError(
            f"В выбранном сечении исходный момент M0={M0/1000:.3f} кН·м не положителен. Эта реализация формул рассчитана на участок с положительным изгибающим моментом. Проверьте координату трещины X={X*1000:.1f} мм, нагрузку и принятую систему знаков; попробуйте выбрать другое сечение."
        )

    if not all(math.isfinite(v) for v in (M0, Q0, N0, Ra, Rb, Ma, q_total)):
        raise ValueError("Расчёт усилий дал NaN или бесконечность. Проверьте исходные данные и единицы измерения.")

    alpha0 = steel.Es / concrete.Eb

    # Начальный подбор диаметра по начальному M0. На каждой итерации
    # диаметр подбирается заново по текущему As0, поэтому может как
    # увеличиваться, так и уменьшаться.
    xi0_initial = M0 / (concrete.Rb * b * h0 ** 2)
    As0_initial = xi0_initial * b * h0 * (concrete.Rb / steel.Ea)
    chosen_d, As = _select_reinforcement(As0_initial, n)

    # Коэффициенты единичных эпюр не зависят от текущего M0 и поэтому
    # вычисляются один раз. Физически они имеют разные единицы.
    dMM, dMQ, dQQ, dMP, dQP = canonical_coefficients(
        L, X, concrete.Eb, I, q_total
    )

    psi_s = steel.psi_s
    psi_b = concrete.psi_b

    current_M0 = M0
    current_Q0 = Q0
    iterations = []
    last_state = None
    converged = False
    # Защита от повторения одного и того же состояния (численный цикл).
    # Округление момента используется только для распознавания повтора,
    # оно не участвует в формулах или критерии сходимости.
    seen_states = {}

    for it in range(1, MAX_ITERATIONS + 1):
        xi, As0, alpha0m, yt, Wpl, Mcrc, Lp = _section_state(
            current_M0,
            b=b, h=h, a=a, h0=h0, n=n,
            chosen_d=chosen_d, As=As, alpha0=alpha0,
            concrete=concrete, steel=steel, N0=N0,
        )

        # На каждой итерации заново выбираем МИНИМАЛЬНЫЙ диаметр,
        # который удовлетворяет текущему As0. Поэтому диаметр может как
        # увеличиваться, так и уменьшаться при изменении изгибающего момента.
        replacement = _select_reinforcement(As0, n)
        if abs(chosen_d - replacement[0]) > 1e-12:
            chosen_d, As = replacement
            # Диаметр изменился — пересчитываем характеристики сечения с ним.
            xi, As0, alpha0m, yt, Wpl, Mcrc, Lp = _section_state(
                current_M0,
                b=b, h=h, a=a, h0=h0, n=n,
                chosen_d=chosen_d, As=As, alpha0=alpha0,
                concrete=concrete, steel=steel, N0=N0,
            )
        else:
            chosen_d, As = replacement

        if Wpl <= 0 or yt <= 0:
            raise ValueError("Некорректная геометрия приведённого сечения: Wpl или yt получились неположительными. Проверьте b, h, защитный слой, количество и диаметр стержней, а также формулы определения центра тяжести и Wpl.")
        if Mcrc <= 0:
            raise ValueError(f"Момент образования трещин Mcrc={Mcrc/1000:.4f} кН·м получился неположительным. Проверьте геометрию приведённого сечения, Rbt, положение арматуры и формулу Mcrc; дальнейший расчёт с таким значением физически неинтерпретируем в этой модели.")

        try:
            s_h0, K, xi1 = _graph_parameters(alpha0m, Lp)
        except ValueError as exc:
            # Сохраняем хотя бы промежуточные величины текущего шага,
            # чтобы пользователь видел, где именно оборвался расчёт.
            iterations.append({
                "i": it, "M0": current_M0, "Q0": current_Q0,
                "xi": xi, "As0": As0, "diameter_mm": chosen_d * 1000.0,
                "As": As, "alpha0": alpha0, "alpha0m": alpha0m,
                "yt": yt, "Wpl": Wpl, "Mcrc": Mcrc, "Lp": Lp,
                "stage_error": str(exc),
            })
            raise IterationCalculationError(str(exc), iterations) from exc
        if s_h0 is None:
            raise ValueError(
                f"Параметр Lp={Lp:.4f} не попадает в область графика Пересыпкина для α0m={alpha0m:.5f}. По имеющейся оцифровке нельзя определить S/h0 и связанные параметры. Это не обязательно ошибка ввода: график может быть недостаточно оцифрован для такого случая. Проверьте исходные данные или дополните таблицу по оригиналу."
            )
        if xi1 <= 1e-12:
            raise ValueError(
                f"По графику получено ξ1={xi1:.6f}, то есть значение равно нулю или слишком близко к нулю. В формуле напряжения бетона σb величина ξ1 стоит в знаменателе, поэтому деление становится невозможным или неустойчивым. Нужна другая допустимая точка графика либо уточнение оцифровки."
            )

        Mb = K * Mcrc

        # σs: Па = Н/м².
        sigma_s = (current_M0 - Mb) / (As0 * (0.98 * h0))

        # Уточненная пользователем формула:
        # σb = [2 σs As0 + Mb/(0.6667*(h-S))] / (ξ1*b*h0), где S=(S/h0)*h0.
        S = s_h0 * h0
        sigma_b = (
            2.0 * sigma_s * As0 + Mb / (0.6667 * (h - S))
        ) / (xi1 * b * h0)

        # Уточненная формула δM*: квадрат относится ко всей сумме в скобках.
        deltaM_star = (
            2.0 * h / (current_M0 * h0)
            * (
                sigma_s * psi_s / steel.Es
                + sigma_b * psi_b / concrete.Eb
            ) ** 2
            - 2.0 * h / (
                concrete.Eb * (Wpl * yt / 1.3)
            )
        )
        derived_values = (sigma_s, sigma_b, deltaM_star, S, Mb)
        if not all(math.isfinite(v) for v in derived_values):
            raise ValueError(
                f"На итерации {it} промежуточные параметры содержат NaN или бесконечность "
                "(напряжения, S, Mb или δM*). Проверьте исходные данные и применимость "
                "диапазона графиков; формулы автоматически не изменялись."
            )

        # Каноническая система по заданию.
        M_new, Q_new = solve_2x2(
            dMM + deltaM_star, dMQ,
            dMQ, dQQ,
            -dMP, -dQP
        )

        if not math.isfinite(M_new) or not math.isfinite(Q_new):
            raise ValueError("Решение канонической системы содержит NaN или бесконечность. Обычно это означает деление на ноль, чрезмерно малый определитель либо некорректный коэффициент. Проверьте параметры графиков и единичные эпюры.")
        if M_new == 0:
            raise ValueError("На итерации получен M_new = 0. Относительное изменение момента и дальнейший расчёт для этого состояния не определены.")

        error = abs((current_M0 - M_new) / current_M0) * 100.0
        if not math.isfinite(error):
            raise ValueError(f"На итерации {it} получена некорректная ошибка сходимости: {error!r}.")

        # Важно: малое изменение момента само по себе недостаточно, если
        # при переходе к M_new меняется подобранный табличный диаметр.
        # Проверяем, останется ли выбранная арматура минимально достаточной
        # для момента следующего состояния. Формулы As0 не меняются.
        xi_new = M_new / (concrete.Rb * b * h0 ** 2)
        As0_new = xi_new * b * h0 * (concrete.Rb / steel.Ea)
        next_d, next_As = _select_reinforcement(As0_new, n)
        diameter_stable = abs(next_d - chosen_d) < 1e-12

        # Сохраняем полный снимок текущего шага ДО проверки сходимости.
        # Поэтому в отчёте будут все реально выполненные итерации, а не только итог.
        iterations.append({
            "i": it,
            "M0": current_M0,
            "Q0": current_Q0,
            "M": M_new,
            "Q": Q_new,
            "error": error,
            "diameter_next_mm": next_d * 1000.0,
            "diameter_stable": diameter_stable,
            "converged_step": error <= MOMENT_TOLERANCE_PERCENT and diameter_stable,
            "xi": xi,
            "As0": As0,
            "diameter_mm": chosen_d * 1000.0,
            "As": As,
            "alpha0": alpha0,
            "alpha0m": alpha0m,
            "yt": yt,
            "Wpl": Wpl,
            "Mcrc": Mcrc,
            "Lp": Lp,
            "S_h0": s_h0,
            "S": S,
            "K": K,
            "xi1": xi1,
            "Mb": Mb,
            "sigma_s": sigma_s,
            "sigma_b": sigma_b,
            "deltaM_star": deltaM_star,
            "deltaMM": dMM,
            "deltaMQ": dMQ,
            "deltaQQ": dQQ,
            "deltaMP": dMP,
            "deltaQP": dQP,
        })

        last_state = {
            "xi": xi, "As0": As0, "alpha0m": alpha0m,
            "yt": yt, "Wpl": Wpl, "Mcrc": Mcrc, "Lp": Lp,
            "S_h0": s_h0, "K": K, "xi1": xi1, "Mb": Mb,
            "sigma_s": sigma_s, "sigma_b": sigma_b,
            "deltaM_star": deltaM_star,
        }

        # Подтверждённая сходимость имеет приоритет над детектором циклов.
        if error <= MOMENT_TOLERANCE_PERCENT and diameter_stable:
            converged = True
            break

        state_key = (round(M_new, 6), round(Q_new, 6), round(next_d * 1000.0, 6))
        if state_key in seen_states:
            first_seen = seen_states[state_key]
            iterations[-1]["cycle_detected"] = True
            return {
                "ok": False,
                "message": (
                    f"Обнаружен повторяющийся итерационный цикл: состояние, похожее на шаг {first_seen}, "
                    f"повторилось на шаге {it}. Расчёт остановлен, чтобы не выдавать несошедшийся "
                    "результат за успешный. Формулы и критерий сходимости не менялись."
                ),
                "iterations": iterations,
            }
        seen_states[state_key] = it

        current_M0 = M_new
        current_Q0 = Q_new

        # Если на следующем шаге M0 станет неположительным, это уже
        # не допустимое состояние для применяемой формулы ξ и графика.
        if current_M0 <= 0:
            return {
                "ok": False,
                "message": (
                    "Итерация привела к M0 ≤ 0. Расчет по данной модели "
                    "в этом исходном состоянии не может быть продолжен."
                ),
                "iterations": iterations,
            }

    if not converged:
        return {
            "ok": False,
            "message": (
                f"Расчёт не сошёлся за {MAX_ITERATIONS} итераций. "
                f"Для успеха требуется изменение момента не более "
                f"{MOMENT_TOLERANCE_PERCENT:g}% и стабильный подобранный диаметр "
                "между текущим и следующим состоянием. Программа не считает "
                "последний шаг подтверждённым результатом. Проверьте исходные "
                "данные, диапазон графиков Пересыпкина и поведение подбора арматуры."
            ),
            "iterations": iterations,
        }

    r = {
        "ok": True,
        "inputs": data,
        "b": b, "h": h, "a": a, "L": L, "X": X,
        "A": A, "I": I, "W": W, "h0": h0,
        "q_self": q_self, "q_total": q_total,
        "Ra": Ra, "Rb": Rb, "Ma": Ma,
        "M0": M0, "Q0": Q0, "N0": N0,
        "alpha0": alpha0,
        "diameter": chosen_d, "diameter_mm": chosen_d * 1000.0,
        "As": As,
        "M": M_new, "Q": Q_new,
        "iterations": iterations,
        "convergence_tolerance_percent": MOMENT_TOLERANCE_PERCENT,
        "max_iterations": MAX_ITERATIONS,
        "reinforcement_stable": True,
        "psi_s": psi_s, "psi_b": psi_b,
        "deltaMM": dMM, "deltaMQ": dMQ, "deltaQQ": dQQ,
        "deltaMP": dMP, "deltaQP": dQP,
    }
    r.update(last_state)
    return r


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
# Оформление в духе приложений Apple: боковая панель навигации,
# карточки, сгруппированные списки ввода, светлая и тёмная темы,
# плавные анимации результатов. Функция calculate() не меняется.

PALETTE = {
    "bg":           ("#F5F5F7", "#000000"),
    "sidebar":      ("#EBEBEF", "#141416"),
    "card":         ("#FFFFFF", "#1C1C1E"),
    "fill":         ("#F2F2F7", "#2C2C2E"),
    "fill2":        ("#E5E5EA", "#3A3A3C"),
    "hover":        ("#DEDEE4", "#2C2C2E"),
    "separator":    ("#E5E5EA", "#38383A"),
    "text":         ("#1D1D1F", "#F5F5F7"),
    "text2":        ("#6E6E73", "#98989D"),
    "text3":        ("#AEAEB2", "#636366"),
    "accent":       ("#0071E3", "#0A84FF"),
    "accent_hover": ("#0077ED", "#409CFF"),
    "green":        ("#34C759", "#30D158"),
    "green_text":   ("#248A3D", "#30D158"),
    "red":          ("#FF3B30", "#FF453A"),
    "orange":       ("#FF9500", "#FF9F0A"),
    "indigo":       ("#5856D6", "#5E5CE6"),
    "teal":         ("#30B0C7", "#40C8E0"),
    "purple":       ("#AF52DE", "#BF5AF2"),
}

# Глифы шрифта Segoe MDL2 Assets (входит в Windows 10/11).
ICON = {
    "calc": "", "diagrams": "", "iter": "", "section": "",
    "protocol": "", "history": "", "save": "", "open": "",
    "clear": "", "copy": "", "export": "", "delete": "",
    "check": "", "warning": "", "info": "", "error": "",
    "rebar": "", "moment": "", "steps": "", "target": "", "stress": "\uE945",
}

MONTHS = ("янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг", "сен", "окт", "ноя", "дек")

rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
rcParams["font.size"] = 10
rcParams["hatch.linewidth"] = 0.7


def blend(c1, c2, t):
    """Линейное смешение двух цветов: t=0 → c1, t=1 → c2."""
    a, b = to_rgb(c1), to_rgb(c2)
    return "#%02X%02X%02X" % tuple(round((x + (y - x) * t) * 255) for x, y in zip(a, b))


def tint(key, t=0.14, base="card"):
    """Пара (светлая, тёмная) — полупрозрачный оттенок цвета на фоне карточки."""
    return tuple(blend(PALETTE[base][i], PALETTE[key][i], t) for i in (0, 1))


def plural(n, forms):
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def make_app_icon(size):
    """Иконка приложения: скруглённый квадрат с градиентом, балка и трещина."""
    s = size * 4
    grad = Image.new("RGBA", (s, s))
    gd = ImageDraw.Draw(grad)
    top, bottom = (100, 210, 255), (0, 102, 220)
    for y in range(s):
        t = y / (s - 1)
        gd.line([(0, y), (s, y)], fill=tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)) + (255,))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, s - 1, s - 1], radius=int(s * 0.23), fill=255)
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    white = (255, 255, 255, 255)
    soft = (255, 255, 255, 205)
    lw = max(2, int(s * 0.02))
    d.line([(s * 0.2, s * 0.25), (s * 0.8, s * 0.25)], fill=soft, width=lw)
    for i in range(5):
        x = s * (0.22 + 0.14 * i)
        d.line([(x, s * 0.25), (x, s * 0.37)], fill=soft, width=lw)
        d.polygon([(x - s * 0.026, s * 0.355), (x + s * 0.026, s * 0.355), (x, s * 0.405)], fill=soft)
    by0, by1 = s * 0.44, s * 0.58
    d.rounded_rectangle([s * 0.14, by0, s * 0.86, by1], radius=int(s * 0.025), fill=white)
    d.polygon([(s * 0.22, by1), (s * 0.15, s * 0.73), (s * 0.29, s * 0.73)], fill=soft)
    d.polygon([(s * 0.78, by1), (s * 0.71, s * 0.73), (s * 0.85, s * 0.73)], fill=soft)
    cx, hgt = s * 0.6, by1 - by0
    pts = [(cx, by1), (cx - s * 0.022, by1 - hgt * 0.35), (cx + s * 0.018, by1 - hgt * 0.62),
           (cx - s * 0.006, by0 + hgt * 0.15)]
    d.line(pts, fill=(255, 59, 48, 255), width=int(s * 0.032), joint="curve")
    return img.resize((size, size), Image.LANCZOS)


def render_level_bar(ratio, px_w, px_h, card_hex, track_hex, dot_hex, ss=3):
    """Горизонтальная шкала: градиент зелёный → красный и точка-маркер (ratio 0…1)."""
    W, H = px_w * ss, px_h * ss
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bh = max(2, int(H * 0.36))
    y0 = (H - bh) // 2
    inset = H // 2
    rect = [inset - bh // 2, y0, W - inset + bh // 2, y0 + bh]
    if ratio is None:
        ImageDraw.Draw(img).rounded_rectangle(rect, radius=bh // 2, fill=track_hex)
        return img.resize((px_w, px_h), Image.LANCZOS)
    stops = [(0.0, (52, 199, 89)), (0.45, (255, 214, 10)), (0.75, (255, 149, 0)), (1.0, (255, 59, 48))]
    row = Image.new("RGBA", (W, 1))
    for x in range(W):
        tt = x / (W - 1)
        for (t0, c0), (t1, c1) in zip(stops[:-1], stops[1:]):
            if tt <= t1:
                f = (tt - t0) / (t1 - t0)
                break
        row.putpixel((x, 0), tuple(round(a + (b - a) * f) for a, b in zip(c0, c1)) + (255,))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layer.paste(row.resize((W, bh)), (0, y0))
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rounded_rectangle(rect, radius=bh // 2, fill=255)
    img.paste(layer, (0, 0), mask)
    d = ImageDraw.Draw(img)
    cx, cy = inset + min(max(ratio, 0.0), 1.0) * (W - 2 * inset), H / 2
    R, r = H * 0.5 - 1, H * 0.28
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=card_hex)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=dot_hex)
    return img.resize((px_w, px_h), Image.LANCZOS)


class App(ctk.CTk):
    NAV = (
        ("calc", "Расчёт", ICON["calc"]),
        ("diagrams", "Эпюры", ICON["diagrams"]),
        ("iter", "Итерации", ICON["iter"]),
        ("section", "Сечение", ICON["section"]),
        ("protocol", "Протокол", ICON["protocol"]),
        ("history", "История", ICON["history"]),
    )
    NUMERIC_KEYS = ("b_mm", "h_mm", "a_mm", "L_m", "q_kN_m", "X_mm", "n")
    FIELD_NAMES = {
        "b_mm": "b — ширина", "h_mm": "h — высота", "a_mm": "a — до центра арматуры",
        "L_m": "L — длина балки", "q_kN_m": "q — нагрузка", "X_mm": "X — положение трещины",
        "n": "n — число стержней",
    }
    ITER_METRICS = ("Момент", "Площадь As₀", "Диаметр", "Сходимость")

    def __init__(self):
        ctk.set_appearance_mode("light")
        super().__init__(fg_color=PALETTE["bg"])
        self.title("Балка с единичной трещиной")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1400, sw - 40), min(900, sh - 90)
        self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 2 - 20)}")
        self.minsize(min(1160, w), min(700, h))
        if sw <= 1440 or sh <= 900:
            self.after(0, lambda: self.state("zoomed"))
        self.vars = {}
        self.entries = {}
        self.result = None
        self.result_inputs = None
        self.current_page = None
        self.iter_steps = []
        self.iter_metric = self.ITER_METRICS[0]
        self._jobs = {}
        self._preview_job = None
        self._stale = set()
        self._mpl_widgets = []
        self._banner_state = ("idle", "", "")
        self._last_ok_banner = None
        self.history_path = Path.home() / ".crack_beam_calculator" / "history.json"
        self._init_fonts()
        self._set_window_icon()
        self._build()
        self._bind_keys()
        self.show_page("calc")
        self.after(80, self._refresh_previews)

    # ------------------------------------------------------------------
    # Базовые элементы
    # ------------------------------------------------------------------

    def _init_fonts(self):
        def f(size, family="Segoe UI", **kw):
            return ctk.CTkFont(family=family, size=size, **kw)
        self.f_large = f(30, "Segoe UI Semibold")
        self.f_title3 = f(17, "Segoe UI Semibold")
        self.f_headline = f(15, "Segoe UI Semibold")
        self.f_body = f(14)
        self.f_body_sb = f(14, "Segoe UI Semibold")
        self.f_caption = f(12)
        self.f_caption_sb = f(12, "Segoe UI Semibold")
        self.f_symbol = f(17, "Cambria", slant="italic")
        self.f_kpi = f(30, "Segoe UI Semibold")
        self.f_ring = f(22, "Segoe UI Semibold")
        self.f_chip = f(19, "Segoe UI Semibold")
        self.f_tile = f(28, "Segoe UI Semibold")
        self.f_tilecap = f(11, "Segoe UI Semibold")
        self.f_icon = f(16, "Segoe MDL2 Assets")
        self.f_icon_s = f(12, "Segoe MDL2 Assets")
        self.f_button = f(16, "Segoe UI Semibold")
        self.f_dot = f(18)

    def _set_window_icon(self):
        try:
            ico = self.history_path.parent / "app_icon.ico"
            ico.parent.mkdir(parents=True, exist_ok=True)
            make_app_icon(256).save(ico, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (256, 256)])
            self.iconbitmap(str(ico))
        except (OSError, tk.TclError):
            pass

    def _is_dark(self):
        return ctk.get_appearance_mode() == "Dark"

    def col(self, key):
        return PALETTE[key][1 if self._is_dark() else 0]

    def _scaling(self):
        return ctk.ScalingTracker.get_widget_scaling(self)

    def _card(self, parent, radius=18, **kw):
        return ctk.CTkFrame(parent, fg_color=PALETTE["card"], corner_radius=radius, **kw)

    def _label(self, parent, text="", font=None, color="text", **kw):
        return ctk.CTkLabel(parent, text=text, font=font or self.f_body, text_color=PALETTE[color], **kw)

    def _pill(self, parent, text, command, glyph=None, primary=False, width=None):
        label = f"{text}" if not glyph else f"{text}"
        btn = ctk.CTkButton(
            parent, text=label, command=command, height=34, corner_radius=17,
            width=width or max(96, 22 + 8 * len(text)),
            font=self.f_body_sb if primary else self.f_body,
            fg_color=PALETTE["accent"] if primary else PALETTE["fill2"],
            hover_color=PALETTE["accent_hover"] if primary else PALETTE["hover"],
            text_color="#FFFFFF" if primary else PALETTE["text"],
        )
        return btn

    def _separator(self, parent, padx=(0, 0)):
        sep = ctk.CTkFrame(parent, height=1, corner_radius=0, fg_color=PALETTE["separator"])
        sep.pack(fill="x", padx=padx)
        return sep

    def _figure(self, parent, height_in):
        fig = Figure(figsize=(4, height_in), dpi=100)
        fig.patch.set_facecolor(self.col("card"))
        canvas = FigureCanvasTkAgg(fig, master=parent)
        widget = canvas.get_tk_widget()
        widget.configure(highlightthickness=0, bd=0, bg=self.col("card"))
        self._mpl_widgets.append(widget)
        return fig, canvas, widget

    def _page_header(self, page, title, subtitle):
        hdr = ctk.CTkFrame(page, fg_color="transparent")
        text = ctk.CTkFrame(hdr, fg_color="transparent")
        text.pack(side="left", fill="x", expand=True)
        self._label(text, title, self.f_large, anchor="w").pack(anchor="w")
        self._label(text, subtitle, self.f_body, "text2", anchor="w", justify="left").pack(anchor="w")
        hdr.actions = ctk.CTkFrame(hdr, fg_color="transparent", width=1, height=1)
        hdr.actions.pack(side="right", anchor="s", pady=(0, 4))
        return hdr

    def _cancel(self, key):
        job = self._jobs.pop(key, None)
        if job:
            try:
                self.after_cancel(job)
            except tk.TclError:
                pass

    def _animate(self, key, duration_ms, frame):
        """Простой аниматор: frame(e) вызывается с e∈[0;1] (ease-out cubic)."""
        self._cancel(key)
        start = time.perf_counter()

        def step():
            t = min(1.0, (time.perf_counter() - start) * 1000.0 / duration_ms)
            frame(1 - (1 - t) ** 3)
            if t < 1.0:
                self._jobs[key] = self.after(15, step)
            else:
                self._jobs.pop(key, None)
        step()

    # ------------------------------------------------------------------
    # Каркас окна
    # ------------------------------------------------------------------

    def _build(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self.content = ctk.CTkFrame(self, fg_color=PALETTE["bg"], corner_radius=0)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)
        self.pages = {}
        builders = {
            "calc": self._build_calc_page, "diagrams": self._build_diagrams_page,
            "iter": self._build_iter_page, "section": self._build_section_page,
            "protocol": self._build_protocol_page, "history": self._build_history_page,
        }
        for key, _, _ in self.NAV:
            page = ctk.CTkFrame(self.content, fg_color="transparent", corner_radius=0)
            page.grid(row=0, column=0, sticky="nsew")
            builders[key](page)
            self.pages[key] = page
        self._build_toast()
        self.refresh_history()

    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=248, corner_radius=0, fg_color=PALETTE["sidebar"])
        sb.grid(row=0, column=0, sticky="nsw")
        sb.grid_propagate(False)
        sb.pack_propagate(False)
        ctk.CTkFrame(self, width=1, corner_radius=0, fg_color=PALETTE["separator"]).grid(row=0, column=0, sticky="nse")

        brand = ctk.CTkFrame(sb, fg_color="transparent")
        brand.pack(fill="x", padx=18, pady=(28, 26))
        self._brand_img = ctk.CTkImage(make_app_icon(132), size=(44, 44))
        ctk.CTkLabel(brand, image=self._brand_img, text="").pack(side="left")
        txt = ctk.CTkFrame(brand, fg_color="transparent")
        txt.pack(side="left", padx=12)
        self._label(txt, "Трещина", self.f_title3, anchor="w", height=22).pack(anchor="w")
        self._label(txt, "Расчёт ЖБ балки", self.f_caption, "text2", anchor="w", height=16).pack(anchor="w")

        self._label(sb, "Разделы", self.f_caption_sb, "text3", anchor="w").pack(fill="x", padx=26, pady=(0, 4))
        self.nav_items = {}
        for key, title, glyph in self.NAV:
            item = ctk.CTkFrame(sb, height=38, corner_radius=10, fg_color="transparent")
            item.pack(fill="x", padx=12, pady=1)
            item.pack_propagate(False)
            ic = ctk.CTkLabel(item, text=glyph, font=self.f_icon, width=26, text_color=PALETTE["accent"])
            ic.pack(side="left", padx=(10, 8))
            lb = ctk.CTkLabel(item, text=title, font=self.f_body, text_color=PALETTE["text"], anchor="w")
            lb.pack(side="left", fill="x", expand=True)
            badge = ctk.CTkLabel(item, text="", font=self.f_caption, text_color=PALETTE["text2"])
            badge.pack(side="right", padx=12)
            for w in (item, ic, lb, badge):
                w.bind("<Button-1>", lambda e, k=key: self.show_page(k))
                w.bind("<Enter>", lambda e, k=key: self._nav_hover(k, True))
                w.bind("<Leave>", lambda e, k=key: self.after(10, lambda: self._nav_hover(k, False)))
            self.nav_items[key] = (item, ic, lb, badge)

        bottom = ctk.CTkFrame(sb, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=16, pady=20)
        self._label(bottom, "Оформление", self.f_caption_sb, "text3", anchor="w").pack(fill="x", padx=10, pady=(0, 6))
        self.appearance = ctk.CTkSegmentedButton(
            bottom, values=["Светлое", "Тёмное"], command=self._set_appearance,
            font=self.f_caption_sb, height=30, corner_radius=9, border_width=2,
            fg_color=PALETTE["fill2"], selected_color=PALETTE["card"],
            selected_hover_color=PALETTE["card"], unselected_color=PALETTE["fill2"],
            unselected_hover_color=PALETTE["hover"], text_color=PALETTE["text"],
        )
        self.appearance.set("Светлое")
        self.appearance.pack(fill="x")
        self._label(bottom, "Ctrl+Enter — рассчитать\nCtrl+1…6 — разделы",
                    self.f_caption, "text3", justify="left", anchor="w").pack(fill="x", padx=10, pady=(14, 0))

    def _nav_hover(self, key, entering):
        item = self.nav_items[key][0]
        if key == self.current_page:
            return
        if not entering:
            try:
                under = self.winfo_containing(*self.winfo_pointerxy())
            except (KeyError, tk.TclError):
                under = None
            w = under
            while w is not None:
                if w is item:
                    return
                w = getattr(w, "master", None)
        item.configure(fg_color=PALETTE["hover"] if entering else "transparent")

    def show_page(self, key):
        self.current_page = key
        self.pages[key].tkraise()
        for k, (item, ic, lb, badge) in self.nav_items.items():
            sel = k == key
            item.configure(fg_color=PALETTE["accent"] if sel else "transparent")
            ic.configure(text_color="#FFFFFF" if sel else PALETTE["accent"])
            lb.configure(text_color="#FFFFFF" if sel else PALETTE["text"], font=self.f_body_sb if sel else self.f_body)
            badge.configure(text_color="#FFFFFF" if sel else PALETTE["text2"])
        if key in self._stale:
            self._redraw(key)
        if hasattr(self, "toast"):
            self.toast.lift()

    def _set_badge(self, key, text):
        self.nav_items[key][3].configure(text=text)

    def _bind_keys(self):
        self.bind("<Control-Return>", lambda e: self.run())
        self.bind("<F5>", lambda e: self.run())
        self.bind("<Control-s>", lambda e: self.save_project_as())
        self.bind("<Control-o>", lambda e: self.open_project())
        for i, (key, _, _) in enumerate(self.NAV, start=1):
            self.bind(f"<Control-Key-{i}>", lambda e, k=key: self.show_page(k))

    # ------------------------------------------------------------------
    # Страница «Расчёт»
    # ------------------------------------------------------------------

    def _build_calc_page(self, page):
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(1, weight=1)
        hdr = self._page_header(page, "Балка с единичной трещиной",
                                "Слева жёсткая заделка, справа шарнирно‑подвижная опора · равномерная нагрузка + собственный вес")
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=(40, 36), pady=(30, 16))
        self._pill(hdr.actions, "Открыть…", self.open_project).pack(side="left", padx=4)
        self._pill(hdr.actions, "Сохранить…", self.save_project_as).pack(side="left", padx=4)
        self._pill(hdr.actions, "Очистить", self.clear).pack(side="left", padx=(4, 0))

        left = ctk.CTkScrollableFrame(
            page, width=384, fg_color="transparent", corner_radius=0,
            scrollbar_button_color=PALETTE["bg"], scrollbar_button_hover_color=PALETTE["fill2"],
        )
        left.grid(row=1, column=0, sticky="ns", padx=(26, 6), pady=(0, 18))
        self._build_inputs(left)

        right = ctk.CTkScrollableFrame(
            page, fg_color="transparent", corner_radius=0,
            scrollbar_button_color=PALETTE["bg"], scrollbar_button_hover_color=PALETTE["fill2"],
        )
        right.grid(row=1, column=1, sticky="nsew", padx=(6, 22), pady=(0, 18))
        self._build_results(right)

    def _group_title(self, parent, text):
        self._label(parent, text, self.f_headline, anchor="w").pack(fill="x", padx=14, pady=(16, 7))

    def _group_card(self, parent):
        """Карточка группы; внутренний отступ сохраняет скруглённые углы видимыми."""
        card = self._card(parent, radius=14)
        card.pack(fill="x", padx=4)
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=4, pady=5)
        return inner

    def _group(self, parent, title, rows):
        self._group_title(parent, title)
        card = self._group_card(parent)
        for i, row in enumerate(rows):
            if i:
                self._separator(card, padx=(50, 0))
            self._input_row(card, *row)
        return card

    def _input_row(self, card, sym, name, key, default, unit):
        row = ctk.CTkFrame(card, fg_color="transparent", height=50)
        row.pack(fill="x")
        row.pack_propagate(False)
        self._label(row, sym, self.f_symbol, "accent", width=24).pack(side="left", padx=(16, 10))
        self._label(row, name, self.f_body, anchor="w").pack(side="left", fill="x", expand=True)
        self._label(row, unit, self.f_body, "text2", width=42, anchor="w").pack(side="right", padx=(8, 10))
        var = tk.StringVar(value=default)
        self.vars[key] = var
        ent = ctk.CTkEntry(row, textvariable=var, width=96, height=32, corner_radius=9, border_width=0,
                           justify="right", font=self.f_body_sb, fg_color=PALETTE["fill"],
                           text_color=PALETTE["text"], border_color=PALETTE["accent"])
        ent.pack(side="right")
        ent.bind("<FocusIn>", lambda e: ent.configure(border_width=2))
        ent.bind("<FocusOut>", lambda e: ent.configure(border_width=0))
        ent.bind("<Return>", lambda e: self.run())
        self.entries[key] = ent
        var.trace_add("write", lambda *_: (self._validate_entry(key), self._schedule_preview()))

    def _build_inputs(self, parent):
        self._group(parent, "Сечение", [
            ("b", "Ширина", "b_mm", "150", "мм"),
            ("h", "Высота", "h_mm", "250", "мм"),
            ("a", "До центра арматуры", "a_mm", "20", "мм"),
        ])
        self._group(parent, "Пролёт и нагрузка", [
            ("L", "Длина балки", "L_m", "4", "м"),
            ("q", "Нагрузка", "q_kN_m", "10", "кН/м"),
            ("X", "Положение трещины", "X_mm", "3400", "мм"),
        ])

        # Армирование: количество стержней (степпер) и класс стали.
        self._group_title(parent, "Армирование")
        card = self._group_card(parent)
        row = ctk.CTkFrame(card, fg_color="transparent", height=50)
        row.pack(fill="x")
        row.pack_propagate(False)
        self._label(row, "n", self.f_symbol, "accent", width=24).pack(side="left", padx=(16, 10))
        self._label(row, "Число стержней", anchor="w").pack(side="left", fill="x", expand=True)
        self.vars["n"] = tk.StringVar(value="3")
        stepper = ctk.CTkFrame(row, fg_color=PALETTE["fill"], corner_radius=9, height=32)
        stepper.pack(side="right", padx=(0, 12))
        for text, delta in (("−", -1), (None, 0), ("+", 1)):
            if text is None:
                ctk.CTkLabel(stepper, textvariable=self.vars["n"], width=34, font=self.f_body_sb,
                             text_color=PALETTE["text"]).pack(side="left")
                continue
            ctk.CTkButton(stepper, text=text, width=36, height=30, corner_radius=8, font=self.f_title3,
                          fg_color="transparent", hover_color=PALETTE["fill2"], text_color=PALETTE["text"],
                          command=lambda d=delta: self._step_n(d)).pack(side="left", padx=1, pady=1)
        self.vars["n"].trace_add("write", lambda *_: self._schedule_preview())
        self._separator(card, padx=(50, 0))
        box = ctk.CTkFrame(card, fg_color="transparent")
        box.pack(fill="x", padx=16, pady=(12, 14))
        top = ctk.CTkFrame(box, fg_color="transparent")
        top.pack(fill="x")
        self._label(top, "Класс арматуры", anchor="w").pack(side="left")
        self.steel_var = tk.StringVar(value="A400")
        self.steel_seg = ctk.CTkSegmentedButton(
            box, values=list(STEELS), variable=self.steel_var, height=32, corner_radius=9, border_width=2,
            font=self.f_caption_sb, fg_color=PALETTE["fill"], selected_color=PALETTE["accent"],
            selected_hover_color=PALETTE["accent_hover"], unselected_color=PALETTE["fill"],
            unselected_hover_color=PALETTE["fill2"], text_color=(PALETTE["text"][0], PALETTE["text"][1]),
            command=lambda v: self._on_material_change(),
        )
        self.steel_seg.pack(fill="x", pady=(10, 6))
        self.steel_note = self._label(box, "", self.f_caption, "text2", anchor="w")
        self.steel_note.pack(fill="x")

        # Бетон: ползунок по классам.
        self._group_title(parent, "Бетон")
        card = self._group_card(parent)
        box = ctk.CTkFrame(card, fg_color="transparent")
        box.pack(fill="x", padx=16, pady=(12, 14))
        top = ctk.CTkFrame(box, fg_color="transparent")
        top.pack(fill="x")
        self._label(top, "Класс бетона", anchor="w").pack(side="left")
        self.concrete_value = self._label(top, "B25", self.f_title3, "accent")
        self.concrete_value.pack(side="right")
        names = list(CONCRETES)
        self.concrete_var = tk.StringVar(value="B25")
        self.concrete_slider = ctk.CTkSlider(
            box, from_=0, to=len(names) - 1, number_of_steps=len(names) - 1, height=20,
            fg_color=PALETTE["fill2"], progress_color=PALETTE["accent"], button_color=PALETTE["accent"],
            button_hover_color=PALETTE["accent_hover"],
            command=lambda v: self.concrete_var.set(names[int(round(v))]),
        )
        self.concrete_slider.set(names.index("B25"))
        self.concrete_slider.pack(fill="x", pady=(10, 2))
        ticks = ctk.CTkFrame(box, fg_color="transparent")
        ticks.pack(fill="x")
        self._label(ticks, names[0], self.f_caption, "text3").pack(side="left")
        self._label(ticks, names[-1], self.f_caption, "text3").pack(side="right")
        self.concrete_note = self._label(box, "", self.f_caption, "text2", anchor="w")
        self.concrete_note.pack(fill="x", pady=(4, 0))
        self.concrete_var.trace_add("write", lambda *_: self._on_material_change())
        self._on_material_change()

        self.calc_button = ctk.CTkButton(
            parent, text="Рассчитать", height=52, corner_radius=14, font=self.f_button,
            fg_color=PALETTE["accent"], hover_color=PALETTE["accent_hover"], text_color="#FFFFFF",
            command=self.run,
        )
        self.calc_button.pack(fill="x", padx=4, pady=(22, 6))
        self._label(parent, "или Ctrl + Enter", self.f_caption, "text3").pack(pady=(0, 10))

    def _step_n(self, delta):
        try:
            n = int(float(self.vars["n"].get()))
        except ValueError:
            n = 3
        self.vars["n"].set(str(min(20, max(1, n + delta))))

    def _on_material_change(self):
        names = list(CONCRETES)
        cname = self.concrete_var.get()
        if cname in CONCRETES:
            c = CONCRETES[cname]
            self.concrete_value.configure(text=cname)
            self.concrete_slider.set(names.index(cname))
            self.concrete_note.configure(
                text=f"Rb = {c.Rb/1e6:.1f} МПа · Rbt = {c.Rbt/1e6:.2f} МПа · Eb = {c.Eb/1e9:.1f} ГПа")
        sname = self.steel_var.get()
        if sname in STEELS:
            s = STEELS[sname]
            self.steel_note.configure(text=f"Rs = {s.Rs/1e6:.0f} МПа · Rsc = {s.Rsc/1e6:.0f} МПа · Es = {s.Es/1e9:.0f} ГПа")
        self._schedule_preview()

    def _validate_entry(self, key):
        raw = self.vars[key].get().strip().replace(",", ".")
        try:
            ok = math.isfinite(float(raw))
        except ValueError:
            ok = False
        self.entries[key].configure(text_color=PALETTE["text"] if ok else PALETTE["red"])

    def _flag_entry(self, key):
        ent = self.entries.get(key)
        if ent is None:
            return
        ent.configure(border_color=PALETTE["red"], border_width=2)
        ent.focus_set()
        self.after(1600, lambda: ent.configure(border_color=PALETTE["accent"],
                                               border_width=2 if self.focus_get() == ent._entry else 0))

    # Градиенты главной карточки результата: (светлая тема, тёмная тема).
    RESULT_GRADIENTS = {
        "idle":  (("#93A7C2", "#556E8E"), ("#2E3D54", "#111925")),
        "ok":    (("#52A8FF", "#1A5AD4"), ("#21508F", "#091936")),
        "warn":  (("#F8B862", "#D06A1E"), ("#6E4619", "#26160A")),
        "error": (("#FF8F80", "#CF3428"), ("#702620", "#290C0A")),
    }

    def _build_results(self, parent):
        # Главная карточка в духе приложения «Погода»: крупное значение
        # момента, статус и мини-график итераций на градиентном фоне.
        self.res_fig = Figure(figsize=(4, 2.4), dpi=100)
        self.res_canvas = FigureCanvasTkAgg(self.res_fig, master=parent)
        self.res_widget = self.res_canvas.get_tk_widget()
        self.res_widget.configure(highlightthickness=0, bd=0, bg=self.col("bg"))
        self.res_widget.pack(fill="x", pady=(0, 14))
        self.res_widget.bind("<Configure>", lambda e: self._schedule_result_redraw(), add="+")
        self._res_progress = 1.0
        self._set_banner("idle")

        # Расчётная схема
        hero = self._card(parent)
        hero.pack(fill="x", pady=(0, 14))
        top = ctk.CTkFrame(hero, fg_color="transparent")
        top.pack(fill="x", padx=20, pady=(16, 0))
        self._label(top, "Расчётная схема", self.f_headline).pack(side="left")
        self.hero_caption = self._label(top, "", self.f_caption, "text2")
        self.hero_caption.pack(side="right")
        self.hero_fig, self.hero_canvas, w = self._figure(hero, 2.15)
        w.pack(fill="x", padx=14, pady=(2, 12))

        # Плитки-виджеты
        grid = ctk.CTkFrame(parent, fg_color="transparent")
        grid.pack(fill="x")
        self.bar_w = 196
        self.tiles = {}
        spec = (
            ("rebar", "Арматура", ICON["rebar"], False),
            ("mcrc", "Трещинообразование", ICON["moment"], False),
            ("crack", "Высота трещины", ICON["warning"], True),
            ("sigma_s", "Напряжение в арматуре", ICON["stress"], True),
            ("sigma_b", "Напряжение в бетоне", ICON["stress"], True),
            ("graph", "Графики Пересыпкина", ICON["diagrams"], False),
        )
        for i, (key, cap, glyph, has_bar) in enumerate(spec):
            r, c = divmod(i, 3)
            grid.grid_columnconfigure(c, weight=1, uniform="tile")
            tile = self._card(grid)
            tile.grid(row=r, column=c, sticky="nsew", padx=(0 if c == 0 else 7, 0 if c == 2 else 7), pady=(0, 14))
            head = ctk.CTkFrame(tile, fg_color="transparent")
            head.pack(fill="x", padx=16, pady=(14, 0))
            ctk.CTkLabel(head, text=glyph, font=self.f_icon_s, text_color=PALETTE["text2"]).pack(side="left")
            ctk.CTkLabel(head, text=cap.upper(), font=self.f_tilecap, text_color=PALETTE["text2"]).pack(side="left", padx=6)
            vr = ctk.CTkFrame(tile, fg_color="transparent")
            vr.pack(fill="x", padx=16)
            val = ctk.CTkLabel(vr, text="—", font=self.f_tile, text_color=PALETTE["text"], height=42)
            val.pack(side="left")
            unit = ctk.CTkLabel(vr, text="", font=self.f_body_sb, text_color=PALETTE["text2"])
            unit.pack(side="left", padx=(5, 0), pady=(10, 0))
            bar_img = None
            if has_bar:
                bar_img = ctk.CTkImage(Image.new("RGBA", (4, 4)), size=(self.bar_w, 14))
                ctk.CTkLabel(tile, image=bar_img, text="").pack(anchor="w", padx=16, pady=(2, 2))
            foot = ctk.CTkLabel(tile, text=" ", font=self.f_caption, text_color=PALETTE["text2"],
                                anchor="w", justify="left", wraplength=220)
            foot.pack(fill="x", padx=16, pady=(2, 14))
            self.tiles[key] = {"val": val, "unit": unit, "foot": foot, "bar": bar_img, "ratio": None}
        self._draw_bars()

    def _set_banner(self, kind, title="", text="", draw=True):
        """Состояние главной карточки: idle / ok / warn / error."""
        if kind == "idle":
            title = title or "Готово к расчёту"
            text = text or ("Схема, сечение и эпюры обновляются прямо во время ввода. "
                            "Нажмите «Рассчитать» или Ctrl + Enter, чтобы запустить итерационный расчёт.")
        self._banner_state = (kind, title, text)
        if kind == "ok":
            self._last_ok_banner = (kind, title, text)
        if draw:
            self._cancel("res")
            self._draw_result_card(1.0)

    def _schedule_result_redraw(self):
        self._cancel("res_redraw")
        self._jobs["res_redraw"] = self.after(40, lambda: self._draw_result_card(self._res_progress))

    def _draw_result_card(self, progress=1.0):
        self._res_progress = progress
        fig = self.res_fig
        fig.clear()
        fig.patch.set_facecolor(self.col("bg"))
        W, H = fig.get_size_inches() * fig.dpi
        s = fig.dpi / 100.0
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        kind, title, text = self._banner_state
        top_c, bottom_c = self.RESULT_GRADIENTS[kind][1 if self._is_dark() else 0]

        box = FancyBboxPatch((1, 1), W - 2, H - 2, boxstyle=f"round,pad=0,rounding_size={18 * s}",
                             fc="none", ec="none")
        ax.add_patch(box)
        yy, xx = np.mgrid[0:1:64j, 0:1:64j]
        im = ax.imshow(np.clip(0.8 * (1 - yy) + 0.2 * xx, 0, 1), extent=(0, W, 0, H), origin="lower",
                       aspect="auto", interpolation="bicubic", vmin=0, vmax=1,
                       cmap=LinearSegmentedColormap.from_list("sky", [top_c, bottom_c]))
        im.set_clip_path(box)
        for cx, cy, rad, alpha in ((W * 0.86, H * 1.1, H * 0.95, 0.08), (W * 1.0, H * 0.1, H * 0.55, 0.06)):
            ax.add_patch(Circle((cx, cy), rad, fc="#FFFFFF", ec="none", alpha=alpha, clip_path=box))

        white = "#FFFFFF"
        pad = 24 * s
        r = self.result
        if kind in ("ok", "warn") and r and r.get("ok"):
            steps = r["iterations"]
            M, M0 = r["M"] / 1000, r["M0"] / 1000
            ax.text(pad, H - pad, "Изгибающий момент в трещине", color=white, alpha=0.9, fontsize=11,
                    fontweight="semibold", va="top")
            num = ax.text(pad - 3 * s, H - pad - 26 * s, f"{M * progress:.3f}", color=white, fontsize=56,
                          fontweight="light", va="top")
            bb = num.get_window_extent(fig.canvas.get_renderer())
            ax.text(bb.x1 + 8 * s, bb.y0 + 14 * s, "кН·м", color=white, alpha=0.9, fontsize=16, va="bottom")
            ax.text(pad, bb.y0 - 6 * s, f"упругий расчёт {M0:.3f}  →  {(M / M0 - 1) * 100:+.1f} %",
                    color=white, alpha=0.85, fontsize=10.5, va="top")
            pill = "✓  Расчёт сошёлся" if kind == "ok" else "Данные изменены — пересчитайте"
            ax.text(W - pad, H - pad, pill, ha="right", va="top", fontsize=10, fontweight="semibold", color=white,
                    bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.95", fc=(1, 1, 1, 0.22), ec="none"))
            it = len(steps)
            ax.text(W - pad, H - pad - 34 * s,
                    f"{it} {plural(it, ('итерация', 'итерации', 'итераций'))} · Δ = {steps[-1]['error']:.2f} %",
                    ha="right", va="top", fontsize=10, color=white, alpha=0.85)

            # Мини-график: упругий момент и результат каждого шага
            vals = np.array([M0] + [z["M"] / 1000 for z in steps])
            labels = ["упругий"] + [f"шаг {z['i']}" for z in steps]
            x0, x1 = max(W * 0.5, bb.x1 + 90 * s), W - pad - 14 * s
            y0, y1 = pad + 24 * s, H - pad - 66 * s
            lo, hi = vals.min(), vals.max()
            span = hi - lo if hi - lo > 1e-12 else 1.0
            px = np.linspace(x0, x1, len(vals))
            py = y0 + (vals - lo) / span * (y1 - y0) if hi - lo > 1e-12 else np.full(len(vals), (y0 + y1) / 2)
            reach = progress * (len(vals) - 1)
            k = int(math.floor(reach))
            lx, ly = list(px[:k + 1]), list(py[:k + 1])
            if k < len(vals) - 1:
                f = reach - k
                lx.append(px[k] + (px[k + 1] - px[k]) * f)
                ly.append(py[k] + (py[k + 1] - py[k]) * f)
            ax.fill_between(lx, y0 - 10 * s, ly, color=white, alpha=0.1, lw=0)
            ax.plot(lx, ly, color=white, lw=2.4, alpha=0.95, solid_capstyle="round", solid_joinstyle="round")
            many = len(vals) > 7
            for j in range(min(k + 1, len(vals))):
                last = j == len(vals) - 1
                if last:
                    ax.scatter([px[j]], [py[j]], s=300, color=white, alpha=0.22, lw=0, zorder=5)
                ax.scatter([px[j]], [py[j]], s=70 if last else 34, color=white, lw=0, zorder=6)
                if not many or j in (0, len(vals) - 1):
                    ax.text(px[j], py[j] + 10 * s, f"{vals[j]:.3f}", ha="center", va="bottom", fontsize=9,
                            color=white, alpha=1.0 if last else 0.85, fontweight="semibold" if last else "normal")
            for j, lab in enumerate(labels):
                if not many or j in (0, len(vals) - 1):
                    ax.text(px[j], pad - 4 * s, lab, ha="center", va="bottom", fontsize=9, color=white, alpha=0.75)
        else:
            caption = {"idle": "Результат", "warn": "Результат", "error": "Расчёт не выполнен"}[kind]
            ax.text(pad, H - pad, caption, color=white, alpha=0.9, fontsize=11, fontweight="semibold", va="top")
            ax.text(pad - 1 * s, H - pad - 22 * s, title, color=white, fontsize=28, fontweight="semibold", va="top")
            chars = max(30, int((W - 2 * pad) / (8.2 * s)))
            lines = textwrap.wrap(text, chars)
            if len(lines) > 5:
                lines = lines[:5]
                lines[-1] = lines[-1].rstrip(" .,;") + "…"
            ax.text(pad, H - pad - 72 * s, "\n".join(lines), color=white, alpha=0.9, fontsize=10.5,
                    va="top", linespacing=1.5)
        self.res_canvas.draw_idle()

    def _set_tile(self, key, value, unit="", foot="", ratio=None):
        t = self.tiles[key]
        t["val"].configure(text=value)
        t["unit"].configure(text=unit)
        t["foot"].configure(text=foot or " ")
        t["ratio"] = ratio

    def _draw_bars(self):
        """Шкалы в плитках, как индикатор УФ-индекса в «Погоде»."""
        px_w = int(round(self.bar_w * self._scaling()))
        px_h = int(round(14 * self._scaling()))
        for t in self.tiles.values():
            if t["bar"] is not None:
                img = render_level_bar(t["ratio"], px_w, px_h, self.col("card"), self.col("fill2"), self.col("text"))
                t["bar"].configure(light_image=img, dark_image=img)

    def _reset_outputs(self):
        for key in self.tiles:
            self._set_tile(key, "—")
        self._draw_bars()

    # ------------------------------------------------------------------
    # Чтение исходных данных
    # ------------------------------------------------------------------

    def _current_inputs(self):
        data = {}
        for key in self.NUMERIC_KEYS:
            raw = self.vars[key].get().strip().replace(",", ".")
            try:
                data[key] = float(raw)
            except ValueError:
                self._flag_entry(key)
                raise ValueError(f"Поле «{self.FIELD_NAMES[key]}» должно содержать число, сейчас там «{raw}».") from None
        data["concrete"] = self.concrete_var.get()
        data["steel"] = self.steel_var.get()
        return data

    def _try_inputs(self):
        try:
            d = {k: float(self.vars[k].get().strip().replace(",", ".")) for k in self.NUMERIC_KEYS}
        except ValueError:
            return None
        if not all(math.isfinite(v) for v in d.values()):
            return None
        d["concrete"] = self.concrete_var.get()
        d["steel"] = self.steel_var.get()
        return d

    def _set_inputs(self, data):
        for key in self.NUMERIC_KEYS:
            value = data[key]
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            self.vars[key].set(str(value))
        self.concrete_var.set(data["concrete"])
        self.steel_var.set(data["steel"])

    def _result_valid(self):
        if not (self.result and self.result.get("ok")):
            return False
        return self._try_inputs() == self.result_inputs

    def _crack_rel(self):
        """Относительная высота трещины S/h по последнему действительному расчёту."""
        if not self._result_valid():
            return None
        r = self.result
        return r["S_h0"] * r["h0"] / r["h"]

    def _schedule_preview(self):
        if self._preview_job:
            self.after_cancel(self._preview_job)
        self._preview_job = self.after(200, self._refresh_previews)

    def _refresh_previews(self):
        self._preview_job = None
        if not hasattr(self, "hero_fig"):
            return
        self._draw_hero()
        self._stale.update({"diagrams", "section"})
        if self.current_page in self._stale:
            self._redraw(self.current_page)
        if self.result and self.result.get("ok"):
            if not self._result_valid() and self._banner_state[0] == "ok":
                self._set_banner("warn", "Исходные данные изменены",
                                 "Показанные результаты относятся к предыдущим данным. Нажмите «Рассчитать», чтобы обновить.")
            elif self._result_valid() and self._banner_state[0] == "warn" and self._last_ok_banner:
                self._set_banner(*self._last_ok_banner)

    def _redraw(self, key):
        self._stale.discard(key)
        if key == "diagrams":
            self._draw_diagrams()
        elif key == "section":
            self._draw_section()
        elif key == "iter":
            self._draw_iter_chart()

    # ------------------------------------------------------------------
    # Рисование: общие помощники matplotlib
    # ------------------------------------------------------------------

    def _mpl_pill(self, ax, x, y, text, fg, bg, dx=0, dy=0, size=9, ha="center", va="center", rotation=0):
        return ax.annotate(
            text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=size,
            color=fg, fontweight="semibold", rotation=rotation, annotation_clip=False, zorder=30,
            bbox=dict(boxstyle="round,pad=0.42,rounding_size=0.85", fc=bg, ec="none"),
        )

    def _style_axes(self, ax):
        ax.set_facecolor(self.col("card"))
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(self.col("separator"))
        ax.tick_params(colors=self.col("text2"), labelsize=9, length=0, pad=7)
        ax.grid(axis="y", color=self.col("separator"), linewidth=0.8)
        ax.set_axisbelow(True)

    def _gradient_area(self, ax, x, y, color, base, alpha=(0.34, 0.0)):
        """Заливка под кривой с вертикальным градиентом прозрачности."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        poly = ax.fill_between(x, base, y, color="none", lw=0)
        paths = poly.get_paths()
        if not paths:
            return
        lo, hi = min(base, np.nanmin(y)), max(base, np.nanmax(y))
        if hi - lo < 1e-12:
            return
        grad = np.zeros((128, 1, 4))
        grad[..., :3] = to_rgb(color)
        grad[:, 0, 3] = np.linspace(alpha[0], alpha[1], 128)
        im = ax.imshow(grad, extent=(x.min(), x.max(), lo, hi), origin="upper", aspect="auto",
                       zorder=2, interpolation="bicubic")
        im.set_clip_path(paths[0], transform=ax.transData)

    def _signed_area(self, ax, x, y, color):
        """Градиентная заливка эпюры: насыщеннее вдали от нулевой линии."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        ymax, ymin = max(y.max(), 0.0), min(y.min(), 0.0)
        for sign in (1, -1):
            part = np.where(sign * y > 0, y, 0.0)
            if not np.any(sign * part > 0):
                continue
            poly = ax.fill_between(x, 0, part, color="none", lw=0)
            paths = poly.get_paths()
            if not paths:
                continue
            grad = np.zeros((128, 1, 4))
            grad[..., :3] = to_rgb(color)
            a = np.linspace(0.36, 0.05, 128)
            grad[:, 0, 3] = a if sign > 0 else a[::-1]
            lo, hi = (0.0, ymax) if sign > 0 else (ymin, 0.0)
            im = ax.imshow(grad, extent=(x.min(), x.max(), lo, hi), origin="upper", aspect="auto",
                           zorder=2, interpolation="bicubic")
            im.set_clip_path(paths[0], transform=ax.transData)

    def _dim(self, ax, p1, p2, color, text, text_color, bg, vertical=False, size=9.5):
        (x1, y1), (x2, y2) = p1, p2
        ax.plot([x1, x2], [y1, y2], color=color, lw=1.0, zorder=6, solid_capstyle="butt")
        ax.plot([x1, x2], [y1, y2], ls="none", marker=(2, 0, 45), markersize=9, mew=1.3, color=color, zorder=6)
        self._mpl_pill(ax, (x1 + x2) / 2, (y1 + y2) / 2, text, text_color, bg, size=size,
                       rotation=90 if vertical else 0)

    def _draw_scheme(self, ax, L, X, q, S_rel=None, reactions=None):
        """Схема балки: заделка A, подвижная опора B, нагрузка q, трещина X."""
        card, txt, t2, t3 = self.col("card"), self.col("text"), self.col("text2"), self.col("text3")
        acc, red, orange, green = self.col("accent"), self.col("red"), self.col("orange"), self.col("green")
        ax.set_facecolor(card)
        ax.axis("off")
        bh = 0.12

        # Нагрузка
        x0, x1 = 0.015 * L, 0.985 * L
        ax.fill_between([x0, x1], bh + 0.02, 0.56, color=acc, alpha=0.07, lw=0, zorder=1)
        ax.plot([x0, x1], [0.56, 0.56], color=acc, lw=1.8, solid_capstyle="round", zorder=5)
        for xx in np.linspace(x0, x1, 19):
            ax.annotate("", xy=(xx, bh + 0.02), xytext=(xx, 0.56), zorder=5,
                        arrowprops=dict(arrowstyle="-|>,head_length=0.42,head_width=0.18", color=acc,
                                        lw=1.1, shrinkA=0, shrinkB=0))
        self._mpl_pill(ax, L / 2, 0.56, f"q = {q:.2f} кН/м", "#FFFFFF", acc, dy=13, va="bottom")

        # Балка с градиентом «бетона» и линией арматуры
        frame = Rectangle((0, -bh), L, 2 * bh, fc="none", ec="none")
        ax.add_patch(frame)
        top_c, bot_c = ("#F1F2F5", "#C5CAD3") if not self._is_dark() else ("#5A5A60", "#2C2C30")
        im = ax.imshow(np.linspace(0, 1, 64).reshape(-1, 1), aspect="auto", origin="lower", zorder=2,
                       cmap=LinearSegmentedColormap.from_list("beam", [bot_c, top_c]),
                       extent=(0, L, -bh, bh), interpolation="bicubic")
        im.set_clip_path(frame)
        ax.add_patch(Rectangle((0, -bh), L, 2 * bh, fc="none", ec=blend(txt, card, 0.5), lw=1.0, zorder=3))
        ax.plot([0.01 * L, 0.99 * L], [-0.068, -0.068], color=orange, lw=1.5, alpha=0.9, zorder=3,
                solid_capstyle="round")

        # Заделка A
        ax.add_patch(Rectangle((-0.034 * L, -0.44), 0.034 * L, 0.88, fc=self.col("fill"), ec=t3, lw=0,
                               hatch="/////", zorder=1))
        ax.plot([0, 0], [-0.44, 0.44], color=txt, lw=2.4, solid_capstyle="round", zorder=4)
        ax.text(-0.017 * L, 0.5, "A", ha="center", va="bottom", fontsize=11, color=t2, fontweight="semibold")

        # Шарнирно-подвижная опора B
        tw = 0.022 * L
        ax.add_patch(Polygon([[L, -bh], [L - tw, -0.35], [L + tw, -0.35]], closed=True, fc=card, ec=txt,
                             lw=1.5, joinstyle="round", zorder=4))
        ax.plot([L - 1.6 * tw, L + 1.6 * tw], [-0.41, -0.41], color=txt, lw=1.5, solid_capstyle="round", zorder=4)
        ax.add_patch(Rectangle((L - 1.6 * tw, -0.52), 3.2 * tw, 0.1, fc="none", ec=t3, lw=0, hatch="/////", zorder=1))
        ax.text(L + 0.04 * L, 0.0, "B", ha="left", va="center", fontsize=11, color=t2, fontweight="semibold")

        # Трещина: растёт от растянутой (нижней) грани
        if 0 < X < L:
            depth = 2 * bh * min(0.95, S_rel if S_rel is not None else 0.55)
            ys = np.linspace(-bh, -bh + depth, 8)
            xs = X + np.array([0, 0.9, -0.7, 1.0, -0.9, 0.6, -0.4, 0.1]) * 0.0042 * L
            ax.plot(xs, ys, color=red, lw=7, alpha=0.18, zorder=6, solid_capstyle="round", solid_joinstyle="round")
            ax.plot(xs, ys, color=red, lw=2.0, zorder=7, solid_capstyle="round", solid_joinstyle="round")
            if reactions is None:
                dim_y = -0.68
                ax.plot([X, X], [-bh - 0.03, dim_y], color=red, lw=0.9, ls=(0, (2, 2.5)), zorder=3)
                ax.plot([0, 0], [-0.47, dim_y - 0.04], color=t3, lw=0.8, zorder=3)
                self._dim(ax, (0, dim_y), (X, dim_y), red, f"X = {X:.2f} м", red, blend(card, red, 0.13))
            else:
                self._mpl_pill(ax, X, -bh, "трещина", red, blend(card, red, 0.13), dy=-16, va="top")

        if reactions is None:
            dim_y = -0.92
            ax.plot([L, L], [-0.53, dim_y - 0.04], color=t3, lw=0.8, zorder=3)
            ax.plot([0, 0], [dim_y + 0.2, dim_y - 0.04], color=t3, lw=0.8, zorder=3)
            self._dim(ax, (0, dim_y), (L, dim_y), t3, f"L = {L:.2f} м", txt, self.col("fill"))
        else:
            Ra, Rb, Ma = reactions
            for xr, y0, val, ha, dx in ((0.0, -0.47, Ra, "left", 8), (L, -0.55, Rb, "right", -8)):
                ax.annotate("", xy=(xr, y0), xytext=(xr, -0.98), zorder=6,
                            arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", color=green,
                                            lw=1.8, shrinkA=0, shrinkB=0))
                self._mpl_pill(ax, xr, -0.84, f"R = {val:.2f} кН", "#FFFFFF", green, dx=dx, ha=ha)
            self._mpl_pill(ax, 0.0, 0.0, f"Mₐ = {Ma:.2f} кН·м", txt, self.col("fill"), dx=-14, ha="right", size=8.5)

        ax.set_xlim(-0.11 * L, 1.08 * L)
        ax.set_ylim(-1.02, 0.98)

    def _hero_data(self, d):
        """Пролёт, координата трещины и полная нагрузка (как в calculate())."""
        L = d["L_m"]
        X = d["X_mm"] / 1000.0
        q_self = 25_000.0 * (d["b_mm"] / 1000.0) * (d["h_mm"] / 1000.0)
        q_total = q_self + d["q_kN_m"] * 1000.0
        return L, X, q_self, q_total

    def _draw_hero(self):
        fig = self.hero_fig
        fig.clear()
        fig.patch.set_facecolor(self.col("card"))
        ax = fig.add_axes([0, 0, 1, 1])
        d = self._try_inputs()
        if d is None or d["L_m"] <= 0 or d["b_mm"] <= 0 or d["h_mm"] <= 0:
            ax.axis("off")
            ax.text(0.5, 0.5, "Введите корректные исходные данные", ha="center", va="center",
                    color=self.col("text2"), fontsize=12, transform=ax.transAxes)
            self.hero_caption.configure(text="")
            self.hero_canvas.draw_idle()
            return
        L, X, q_self, q_total = self._hero_data(d)
        self._draw_scheme(ax, L, X, q_total / 1000.0, self._crack_rel())
        note = "" if 0 < X < L else "  ·  трещина вне пролёта"
        self.hero_caption.configure(
            text=f"q = {d['q_kN_m']:g} + собственный вес {q_self/1000:.2f} кН/м{note}")
        self.hero_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Страница «Эпюры»
    # ------------------------------------------------------------------

    def _build_diagrams_page(self, page):
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(2, weight=1)
        hdr = self._page_header(page, "Эпюры M и Q",
                                "Статический расчёт по введённым данным · строится сразу, без запуска итераций")
        hdr.grid(row=0, column=0, sticky="ew", padx=(40, 36), pady=(30, 16))
        chips = ctk.CTkFrame(page, fg_color="transparent")
        chips.grid(row=1, column=0, sticky="ew", padx=(36, 36), pady=(0, 14))
        self.diag_chips = {}
        spec = (("Ra", "Реакция A", "кН"), ("Rb", "Реакция B", "кН"), ("Ma", "Момент в заделке", "кН·м"),
                ("Mmax", "Максимум M", "кН·м"), ("MX", "M в трещине", "кН·м"), ("QX", "Q в трещине", "кН"))
        for i, (key, cap, unit) in enumerate(spec):
            chips.grid_columnconfigure(i, weight=1, uniform="chip")
            ch = self._card(chips, radius=14)
            ch.grid(row=0, column=i, sticky="nsew", padx=4)
            self._label(ch, cap, self.f_caption, "text2", anchor="w").pack(fill="x", padx=14, pady=(10, 0))
            row = ctk.CTkFrame(ch, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=(0, 10))
            v = self._label(row, "—", self.f_chip)
            v.pack(side="left")
            self._label(row, unit, self.f_caption, "text2").pack(side="left", padx=(4, 0), pady=(5, 0))
            self.diag_chips[key] = v
        card = self._card(page)
        card.grid(row=2, column=0, sticky="nsew", padx=(40, 40), pady=(0, 26))
        self.diag_fig, self.diag_canvas, w = self._figure(card, 2)
        w.pack(fill="both", expand=True, padx=16, pady=16)
        self._stale.add("diagrams")

    def _draw_diagrams(self):
        fig = self.diag_fig
        fig.clear()
        card, t2, t3, red = self.col("card"), self.col("text2"), self.col("text3"), self.col("red")
        fig.patch.set_facecolor(card)
        d = self._try_inputs()
        valid = d is not None and d["L_m"] > 0 and d["b_mm"] > 0 and d["h_mm"] > 0 and d["q_kN_m"] >= 0
        if valid:
            L, X, _, q_N = self._hero_data(d)
            valid = 0 < X < L
        if not valid:
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            ax.text(0.5, 0.5, "Введите корректные L, X, q, b и h,\nчтобы построить эпюры", ha="center",
                    va="center", color=t2, fontsize=13, linespacing=1.6, transform=ax.transAxes)
            for v in self.diag_chips.values():
                v.configure(text="—")
            self.diag_canvas.draw_idle()
            return

        _, _, _, Ra, Rb, Ma = beam_actions(L, q_N, 0.0)
        xs = np.linspace(0.0, L, 600)
        Ms = np.array([beam_actions(L, q_N, x)[0] for x in xs]) / 1000.0
        Qs = np.array([beam_actions(L, q_N, x)[1] for x in xs]) / 1000.0
        MX, QX = (v / 1000.0 for v in beam_actions(L, q_N, X)[:2])
        x_mmax = min(max(Ra / q_N, 0.0), L) if q_N > 0 else 0.0
        Mmax = beam_actions(L, q_N, x_mmax)[0] / 1000.0
        for key, val in (("Ra", Ra / 1000), ("Rb", Rb / 1000), ("Ma", Ma / 1000), ("Mmax", Mmax),
                         ("MX", MX), ("QX", QX)):
            self.diag_chips[key].configure(text=f"{val:.2f}")

        S_rel = self._crack_rel()
        gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1, 1], hspace=0.5,
                              left=0.065, right=0.975, top=0.97, bottom=0.07)
        ax_s = fig.add_subplot(gs[0])
        self._draw_scheme(ax_s, L, X, q_N / 1000.0, S_rel, reactions=(Ra / 1000, Rb / 1000, Ma / 1000))
        ax_m = fig.add_subplot(gs[1])
        ax_q = fig.add_subplot(gs[2])
        acc, purple = self.col("accent"), self.col("purple")

        def finish(ax, Y, color, title, unit, labels):
            self._style_axes(ax)
            ax.spines["bottom"].set_visible(False)
            self._signed_area(ax, xs, Y, color)
            ax.plot([0, L], [0, 0], color=t3, lw=1.0, zorder=3)
            ax.plot(xs, Y, color=color, lw=2.4, solid_capstyle="round", zorder=5)
            ax.plot([0, 0], [0, Y[0]], color=color, lw=2.4, zorder=5)
            ax.plot([L, L], [0, Y[-1]], color=color, lw=2.4, zorder=5)
            ax.axvline(X, color=red, lw=1.0, ls=(0, (3, 3)), zorder=4)
            rng = max(abs(Y.max()), abs(Y.min()), 1e-9)
            for xr, yr, txt in labels:
                ax.scatter([xr], [yr], s=46, facecolor=card, edgecolor=color, lw=2, zorder=7)
                if xr <= 1e-9 or xr >= L - 1e-9:
                    # У опор подпись ставится сбоку, чтобы не залезать на заголовок.
                    left = xr <= 1e-9
                    self._mpl_pill(ax, xr, yr, txt, "#FFFFFF", color, dx=10 if left else -10,
                                   ha="left" if left else "right", size=8.5)
                    continue
                up = yr >= 0
                self._mpl_pill(ax, xr, yr, txt, "#FFFFFF", color, dy=11 if up else -11,
                               va="bottom" if up else "top", size=8.5)
            ax.set_ylim(min(Y.min(), 0) - 0.5 * rng, max(Y.max(), 0) + 0.62 * rng)
            ax.set_xlim(-0.11 * L, 1.08 * L)
            ax.set_xticks(np.linspace(0, L, 5))
            ax.xaxis.set_major_formatter(lambda v, _: f"{v:.2f} м")
            ax.text(0.0, 1.04, title, transform=ax.transAxes, fontsize=11.5, fontweight="semibold",
                    color=self.col("text"), va="bottom")
            ax.text(1.0, 1.04, unit, transform=ax.transAxes, fontsize=9, color=t2, va="bottom", ha="right")

        finish(ax_m, Ms, acc, "Изгибающий момент M", "кН·м  ·  положительный вверх",
               [(0.0, Ma / 1000, f"{Ma/1000:.2f}"), (x_mmax, Mmax, f"max {Mmax:.2f}"), (X, MX, f"M(X) = {MX:.2f}")])
        finish(ax_q, Qs, purple, "Поперечная сила Q", "кН",
               [(0.0, Ra / 1000, f"{Ra/1000:.2f}"), (X, QX, f"Q(X) = {QX:.2f}"), (L, -Rb / 1000, f"{-Rb/1000:.2f}")])
        ax_m.tick_params(labelbottom=False)
        self.diag_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Страница «Итерации»
    # ------------------------------------------------------------------

    def _build_iter_page(self, page):
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=3)
        page.grid_rowconfigure(2, weight=2)
        hdr = self._page_header(page, "Ход итераций",
                                f"Критерий: изменение момента ≤ {MOMENT_TOLERANCE_PERCENT:g} % и стабильный диаметр · "
                                f"не более {MAX_ITERATIONS} шагов")
        hdr.grid(row=0, column=0, sticky="ew", padx=(40, 36), pady=(30, 16))
        self.iter_seg = ctk.CTkSegmentedButton(
            hdr.actions, values=list(self.ITER_METRICS), command=self._set_iter_metric, height=32,
            corner_radius=9, border_width=2, font=self.f_caption_sb, fg_color=PALETTE["fill2"],
            selected_color=PALETTE["card"], selected_hover_color=PALETTE["card"],
            unselected_color=PALETTE["fill2"], unselected_hover_color=PALETTE["hover"],
            text_color=PALETTE["text"],
        )
        self.iter_seg.set(self.iter_metric)
        self.iter_seg.pack()
        card = self._card(page)
        card.grid(row=1, column=0, sticky="nsew", padx=40, pady=(0, 14))
        self.iter_fig, self.iter_canvas, w = self._figure(card, 2)
        w.pack(fill="both", expand=True, padx=16, pady=14)
        table_card = self._card(page)
        table_card.grid(row=2, column=0, sticky="nsew", padx=40, pady=(0, 26))
        self.iter_table = ctk.CTkScrollableFrame(table_card, fg_color="transparent", corner_radius=0,
                                                 scrollbar_button_color=PALETTE["card"],
                                                 scrollbar_button_hover_color=PALETTE["fill2"])
        self.iter_table.pack(fill="both", expand=True, padx=10, pady=8)
        self._fill_iter_table([])
        self._stale.add("iter")

    def _set_iter_metric(self, value):
        self.iter_metric = value
        self._draw_iter_chart()

    def _set_iterations(self, steps):
        self.iter_steps = list(steps or [])
        self._set_badge("iter", str(len(self.iter_steps)) if self.iter_steps else "")
        self._fill_iter_table(self.iter_steps)
        self._stale.add("iter")
        if self.current_page == "iter":
            self._redraw("iter")

    def _draw_iter_chart(self):
        fig = self.iter_fig
        fig.clear()
        card, t2, t3 = self.col("card"), self.col("text2"), self.col("text3")
        fig.patch.set_facecolor(card)
        ax = fig.add_axes([0.07, 0.14, 0.9, 0.76])
        self._style_axes(ax)
        steps = self.iter_steps
        if not steps:
            ax.axis("off")
            ax.text(0.5, 0.5, "Здесь появится ход итераций после расчёта", ha="center", va="center",
                    color=t2, fontsize=13, transform=ax.transAxes)
            self.iter_canvas.draw_idle()
            return
        x = np.array([z.get("i", k + 1) for k, z in enumerate(steps)], dtype=float)
        nan = float("nan")
        refs = []
        metric = self.iter_metric
        if metric == "Момент":
            y = np.array([z.get("M", nan) / 1000 for z in steps])
            color, unit, fmt = self.col("accent"), "M, кН·м", "{:.3f}"
            refs.append((steps[0]["M0"] / 1000, "M₀ упругого расчёта"))
        elif metric == "Площадь As₀":
            y = np.array([z.get("As0", nan) * 1e6 for z in steps])
            color, unit, fmt = self.col("teal"), "As₀, мм²", "{:.1f}"
        elif metric == "Диаметр":
            y = np.array([z.get("diameter_mm", nan) for z in steps])
            color, unit, fmt = self.col("orange"), "Ø, мм", "Ø{:.0f}"
        else:
            y = np.array([z.get("error", nan) for z in steps])
            color, unit, fmt = self.col("green"), "Изменение момента, %", "{:.2f} %"
            refs.append((MOMENT_TOLERANCE_PERCENT, f"допуск {MOMENT_TOLERANCE_PERCENT:g} %"))
        mask = np.isfinite(y)
        if not mask.any():
            ax.axis("off")
            ax.text(0.5, 0.5, "Для этой величины нет данных: расчёт остановился раньше",
                    ha="center", va="center", color=t2, fontsize=12, transform=ax.transAxes)
            self.iter_canvas.draw_idle()
            return
        xv, yv = x[mask], y[mask]
        vals = list(yv) + [r[0] for r in refs]
        lo, hi = min(vals), max(vals)
        rng = max(hi - lo, abs(hi) * 0.08, 1e-9)
        lo, hi = lo - 0.35 * rng, hi + 0.45 * rng
        if metric == "Сходимость":
            lo = min(0.0, lo)
            ax.axhspan(lo, MOMENT_TOLERANCE_PERCENT, color=self.col("green"), alpha=0.08, lw=0, zorder=0)
        for value, label in refs:
            ax.axhline(value, color=t3, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.text(x.min() - 0.45, value, f" {label}", color=t2, fontsize=8.5, va="bottom", ha="left")
        if len(xv) > 1:
            self._gradient_area(ax, xv, yv, color, lo)
        if metric == "Диаметр":
            ax.step(xv, yv, where="mid", color=color, lw=2.6, zorder=5)
        else:
            ax.plot(xv, yv, color=color, lw=2.6, zorder=5, solid_capstyle="round", solid_joinstyle="round")
        ax.scatter(xv[:-1], yv[:-1], s=52, facecolor=card, edgecolor=color, lw=2.2, zorder=6)
        ax.scatter(xv[-1:], yv[-1:], s=420, color=color, alpha=0.15, lw=0, zorder=6)
        ax.scatter(xv[-1:], yv[-1:], s=95, color=color, edgecolor=card, lw=2, zorder=7)
        for xx, yy in zip(xv[:-1], yv[:-1]):
            ax.annotate(fmt.format(yy), (xx, yy), xytext=(0, 11), textcoords="offset points", ha="center",
                        fontsize=9, color=t2, zorder=8)
        self._mpl_pill(ax, xv[-1], yv[-1], fmt.format(yv[-1]), "#FFFFFF", color, dy=15, va="bottom", size=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Шаг {int(i)}" for i in x])
        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
        ax.set_ylim(lo, hi)
        ax.text(0.0, 1.05, unit, transform=ax.transAxes, color=t2, fontsize=9.5, va="bottom")
        self.iter_canvas.draw_idle()

    def _fill_iter_table(self, steps):
        for w in self.iter_table.winfo_children():
            w.destroy()
        headers = ("Шаг", "M₀, кН·м", "M, кН·м", "Δ, %", "As₀, мм²", "Ø, мм", "σs, МПа", "σb, МПа", "Итог")
        for j, text in enumerate(headers):
            self.iter_table.grid_columnconfigure(j, weight=1)
            self._label(self.iter_table, text, self.f_caption_sb, "text2",
                        anchor="w" if j in (0, 8) else "e").grid(row=0, column=j, sticky="ew", padx=10, pady=(8, 6))
        if not steps:
            self._label(self.iter_table, "Пока пусто — выполните расчёт", self.f_body, "text3").grid(
                row=2, column=0, columnspan=len(headers), pady=26)
            return

        def num(z, key, scale=1.0, fmt="{:.3f}"):
            v = z.get(key)
            return fmt.format(v * scale) if isinstance(v, (int, float)) and math.isfinite(v) else "—"

        for i, z in enumerate(steps):
            r = 2 * i + 1
            ctk.CTkFrame(self.iter_table, height=1, corner_radius=0, fg_color=PALETTE["separator"]).grid(
                row=r, column=0, columnspan=len(headers), sticky="ew", padx=6)
            if z.get("stage_error"):
                status, color = "остановлен", "red"
            elif z.get("converged_step"):
                status, color = "сошлось", "green_text"
            elif z.get("cycle_detected"):
                status, color = "цикл", "orange"
            else:
                status, color = "далее", "text2"
            cells = (str(z.get("i", i + 1)), num(z, "M0", 1e-3), num(z, "M", 1e-3), num(z, "error", 1, "{:.2f}"),
                     num(z, "As0", 1e6, "{:.1f}"), num(z, "diameter_mm", 1, "{:.0f}"),
                     num(z, "sigma_s", 1e-6, "{:.1f}"), num(z, "sigma_b", 1e-6, "{:.2f}"))
            for j, text in enumerate(cells):
                self._label(self.iter_table, text, self.f_body_sb if j == 0 else self.f_body,
                            anchor="w" if j == 0 else "e").grid(row=r + 1, column=j, sticky="ew", padx=10, pady=5)
            self._label(self.iter_table, "●  " + status, self.f_caption_sb, color, anchor="w").grid(
                row=r + 1, column=8, sticky="ew", padx=10, pady=5)

    # ------------------------------------------------------------------
    # Страница «Сечение»
    # ------------------------------------------------------------------

    def _build_section_page(self, page):
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        hdr = self._page_header(page, "Поперечное сечение",
                                "a — расстояние от растянутой грани до центра арматуры, h₀ = h − a")
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=(40, 36), pady=(30, 16))
        card = self._card(page)
        card.grid(row=1, column=0, sticky="nsew", padx=(40, 7), pady=(0, 26))
        self.section_fig, self.section_canvas, w = self._figure(card, 2)
        w.pack(fill="both", expand=True, padx=16, pady=16)
        info = self._card(page)
        info.grid(row=1, column=1, sticky="nsew", padx=(7, 40), pady=(0, 26))
        info.configure(width=360)
        self._label(info, "Характеристики", self.f_headline, anchor="w").pack(fill="x", padx=20, pady=(18, 8))
        self.section_rows = {}
        rows = (("b", "Ширина b"), ("h", "Высота h"), ("a", "До центра арматуры a"), ("h0", "Рабочая высота h₀"),
                ("A", "Площадь бетона A"), ("I", "Момент инерции I"), ("rebar", "Армирование"),
                ("As", "Площадь арматуры As"), ("mu", "Процент армирования μ"),
                ("alpha0", "α₀ = Es / Eb"), ("Mcrc", "Mcrc"), ("mat", "Материалы"))
        for i, (key, label) in enumerate(rows):
            if i:
                self._separator(info, padx=(20, 20))
            r = ctk.CTkFrame(info, fg_color="transparent", height=36)
            r.pack(fill="x", padx=20)
            r.pack_propagate(False)
            self._label(r, label, self.f_caption, "text2", anchor="w").pack(side="left")
            v = self._label(r, "—", self.f_body_sb, anchor="e")
            v.pack(side="right")
            self.section_rows[key] = v
        self._stale.add("section")

    def _draw_section(self):
        fig = self.section_fig
        fig.clear()
        card, txt, t2, t3 = self.col("card"), self.col("text"), self.col("text2"), self.col("text3")
        orange, acc = self.col("orange"), self.col("accent")
        fig.patch.set_facecolor(card)
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        ax.set_facecolor(card)
        ax.axis("off")
        d = self._try_inputs()
        try:
            if d is None:
                raise ValueError
            b, h, a = d["b_mm"], d["h_mm"], d["a_mm"]
            n = int(d["n"])
            if not all(v > 0 for v in (b, h, a)) or n < 1 or a >= h / 2 or (n > 1 and 2 * a >= b):
                raise ValueError
        except (ValueError, TypeError):
            ax.text(0.5, 0.5, "Введите корректные b, h, a и n", ha="center", va="center", color=t2,
                    fontsize=13, transform=ax.transAxes)
            for v in self.section_rows.values():
                v.configure(text="—")
            self.section_canvas.draw_idle()
            return
        res = self.result if self._result_valid() else None
        dmm = res["diameter_mm"] if res else 0.0

        # Бетон: градиент и лёгкая «зернистость» заполнителя
        rect = Rectangle((0, 0), b, h, fc="none", ec="none")
        ax.add_patch(rect)
        top_c, bot_c = ("#F3F4F6", "#D3D7DE") if not self._is_dark() else ("#4E4E54", "#2E2E32")
        im = ax.imshow(np.linspace(0, 1, 64).reshape(-1, 1), extent=(0, b, 0, h), origin="lower", aspect="auto",
                       cmap=LinearSegmentedColormap.from_list("conc", [bot_c, top_c]), zorder=1,
                       interpolation="bicubic")
        im.set_clip_path(rect)
        rng = np.random.default_rng(7)
        cnt = int(min(900, max(120, b * h / 90)))
        pts = rng.random((cnt, 2)) * [b, h]
        ax.scatter(pts[:, 0], pts[:, 1], s=rng.random(cnt) * 5 + 0.6, color=blend(card, txt, 0.3), alpha=0.35,
                   lw=0, zorder=2)
        ax.add_patch(Rectangle((0, 0), b, h, fc="none", ec=blend(txt, card, 0.25), lw=1.6, zorder=5))

        # Арматура
        radius = dmm / 2 if dmm else max(3.0, min(b, h) * 0.03)
        xs = np.linspace(a, b - a, n) if n > 1 else [b / 2]
        ax.plot([0, b], [a, a], color=orange, lw=0.8, ls=(0, (3, 3)), alpha=0.7, zorder=3)
        for xx in xs:
            if dmm:
                ax.add_patch(Circle((xx, a), radius, fc=blend(orange, "#000000", 0.28), ec="none", zorder=6))
                ax.add_patch(Circle((xx, a), radius * 0.82, fc=orange, ec="none", zorder=6))
                ax.add_patch(Circle((xx - radius * 0.3, a + radius * 0.3), radius * 0.32, fc="#FFFFFF",
                                    alpha=0.55, ec="none", zorder=7))
            else:
                ax.add_patch(Circle((xx, a), radius, fc="none", ec=t2, lw=1.2, ls=(0, (2, 2)), zorder=6))

        # Размеры
        off = 0.12 * max(b, h)
        fill = self.col("fill")
        ax.plot([0, 0], [h + 0.15 * off, h + 0.75 * off], color=t3, lw=0.8)
        ax.plot([b, b], [h + 0.15 * off, h + 0.75 * off], color=t3, lw=0.8)
        self._dim(ax, (0, h + 0.6 * off), (b, h + 0.6 * off), t2, f"b = {b:g} мм", txt, fill)
        ax.plot([b + 0.15 * off, b + 0.75 * off], [0, 0], color=t3, lw=0.8)
        ax.plot([b + 0.15 * off, b + 0.75 * off], [h, h], color=t3, lw=0.8)
        self._dim(ax, (b + 0.6 * off, 0), (b + 0.6 * off, h), t2, f"h = {h:g} мм", txt, fill, vertical=True)
        ax.plot([-0.15 * off, -1.45 * off], [0, 0], color=t3, lw=0.8)
        ax.plot([-0.15 * off, -1.45 * off], [h, h], color=t3, lw=0.8)
        ax.plot([-0.15 * off, -0.75 * off], [a, a], color=t3, lw=0.8)
        self._dim(ax, (-0.6 * off, 0), (-0.6 * off, a), orange, f"a = {a:g}", "#FFFFFF", orange, vertical=True, size=8.5)
        self._dim(ax, (-1.3 * off, a), (-1.3 * off, h), acc, f"h₀ = {h - a:g} мм", "#FFFFFF", acc, vertical=True)
        label = f"{n} × Ø{dmm:g} мм" if dmm else f"n = {n} · диаметр подбирается при расчёте"
        self._mpl_pill(ax, b / 2, -0.55 * off, label, "#FFFFFF" if dmm else txt, orange if dmm else fill, size=10)
        # Невидимые угловые точки задают рамку чертежа; масштаб 1:1 по осям.
        ax.plot([-1.9 * off, b + 1.2 * off], [-1.1 * off, h + 1.2 * off], alpha=0)
        ax.margins(0)
        ax.set_aspect("equal", adjustable="datalim")

        # Таблица характеристик
        conc, steel = CONCRETES.get(d["concrete"]), STEELS.get(d["steel"])
        rows = self.section_rows
        rows["b"].configure(text=f"{b:g} мм")
        rows["h"].configure(text=f"{h:g} мм")
        rows["a"].configure(text=f"{a:g} мм")
        rows["h0"].configure(text=f"{h - a:g} мм")
        rows["A"].configure(text=f"{b * h / 100:.1f} см²")
        rows["I"].configure(text=f"{b * h ** 3 / 12 / 1e4:,.0f} см⁴".replace(",", " "))
        if res:
            rows["rebar"].configure(text=f"{n} × Ø{dmm:g} {d['steel']}")
            rows["As"].configure(text=f"{res['As'] * 1e6:.1f} мм²")
            rows["mu"].configure(text=f"{res['As'] / (res['b'] * res['h0']) * 100:.3f} %")
            rows["Mcrc"].configure(text=f"{res['Mcrc'] / 1000:.3f} кН·м")
        else:
            for k in ("rebar", "As", "mu", "Mcrc"):
                rows[k].configure(text="после расчёта" if k == "rebar" else "—")
        rows["alpha0"].configure(text=f"{steel.Es / conc.Eb:.3f}" if conc and steel else "—")
        rows["mat"].configure(text=f"{d['concrete']} · {d['steel']}")
        self.section_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Страница «Протокол»
    # ------------------------------------------------------------------

    def _build_protocol_page(self, page):
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        hdr = self._page_header(page, "Протокол расчёта", "Все формулы, промежуточные величины и полный ход итераций")
        hdr.grid(row=0, column=0, sticky="ew", padx=(40, 36), pady=(30, 16))
        self._pill(hdr.actions, "Копировать", self._copy_protocol).pack(side="left", padx=4)
        self._pill(hdr.actions, "Сохранить .txt…", self._export_protocol).pack(side="left", padx=(4, 0))
        card = self._card(page)
        card.grid(row=1, column=0, sticky="nsew", padx=40, pady=(0, 26))
        self.protocol = ctk.CTkTextbox(card, fg_color=PALETTE["card"], text_color=PALETTE["text"], wrap="word",
                                       corner_radius=0, border_width=0, font=self.f_body,
                                       scrollbar_button_color=PALETTE["fill2"],
                                       scrollbar_button_hover_color=PALETTE["text3"])
        self.protocol.pack(fill="both", expand=True, padx=18, pady=16)
        self._config_protocol_tags()
        self._render_protocol("Протокол пуст\n\nЗдесь появится подробный расчёт после нажатия кнопки «Рассчитать».")

    def _config_protocol_tags(self):
        tb = self.protocol._textbox
        c = self.col
        tb.configure(padx=10, pady=6, spacing1=1, spacing3=1,
                     selectbackground=blend(c("card"), c("accent"), 0.28), selectforeground=c("text"))
        tb.tag_configure("h1", font=("Segoe UI Semibold", 20), foreground=c("text"), spacing1=4, spacing3=12)
        tb.tag_configure("h2", font=("Segoe UI Semibold", 13), foreground=c("text"), spacing1=16, spacing3=6)
        tb.tag_configure("iter", font=("Segoe UI Semibold", 13), foreground=c("accent"), spacing1=18, spacing3=6)
        tb.tag_configure("h3", font=("Segoe UI Semibold", 10), foreground=c("text"), spacing1=8, spacing3=2,
                         lmargin1=14, lmargin2=14)
        tb.tag_configure("kv", font=("Consolas", 11), foreground=c("text"), lmargin1=14, lmargin2=14)
        tb.tag_configure("note", font=("Segoe UI", 10), foreground=c("text2"), lmargin1=30, lmargin2=30)
        tb.tag_configure("result", font=("Consolas", 10), foreground=c("text"), lmargin1=30, lmargin2=30)
        tb.tag_configure("body", font=("Segoe UI", 11), foreground=c("text"), lmargin1=0, lmargin2=0)
        tb.tag_configure("error", font=("Segoe UI Semibold", 13), foreground=c("red"), spacing1=6, spacing3=6)
        tb.tag_configure("rule", font=("Segoe UI", 1), background=c("separator"), spacing1=0, spacing3=0)

    def _render_protocol(self, text):
        self._protocol_raw = text
        tb = self.protocol._textbox
        tb.configure(state="normal")
        tb.delete("1.0", "end")
        first = True
        result_prefixes = ("Результат:", "Принято", "M = ", "относительное", "проверка", "итог шага", "диаметр для")
        for line in text.split("\n"):
            s = line.strip()
            if s and set(s) <= set("=-"):
                tb.insert("end", "\n", "rule")
                continue
            if first and s:
                tag, first = "h1", False
            elif s.startswith("ИТЕРАЦИЯ №"):
                tag = "iter"
            elif s.startswith(("РАСЧЁТ ОСТАНОВЛЕН", "РАСЧЕТ НЕ ВЫПОЛНЕН")):
                tag = "error"
            elif re.match(r"^\d+\.\s", s) or (s.isupper() and len(s) > 3):
                tag = "h2"
            elif re.match(r"^\d\)\s", s):
                tag = "h3"
            elif line.startswith("   "):
                tag = "result" if s.startswith(result_prefixes) else "note"
            elif " = " in s:
                tag = "kv"
            else:
                tag = "body"
            tb.insert("end", s + "\n", tag)

    def _copy_protocol(self):
        self.clipboard_clear()
        self.clipboard_append(getattr(self, "_protocol_raw", ""))
        self.show_toast("Протокол скопирован", ICON["copy"])

    def _export_protocol(self):
        path = filedialog.asksaveasfilename(title="Сохранить протокол", defaultextension=".txt",
                                            filetypes=(("Текст (*.txt)", "*.txt"), ("Все файлы", "*.*")))
        if not path:
            return
        try:
            Path(path).write_text(getattr(self, "_protocol_raw", ""), encoding="utf-8")
            self.show_toast("Протокол сохранён", ICON["save"])
        except OSError as exc:
            messagebox.showerror("Ошибка сохранения", str(exc))

    # ------------------------------------------------------------------
    # Страница «История»
    # ------------------------------------------------------------------

    def _build_history_page(self, page):
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        hdr = self._page_header(page, "История", "Последние расчёты · нажмите «Открыть», чтобы вернуть исходные данные")
        hdr.grid(row=0, column=0, sticky="ew", padx=(40, 36), pady=(30, 16))
        self._pill(hdr.actions, "Обновить", self.refresh_history).pack(side="left")
        self.history_list = ctk.CTkScrollableFrame(page, fg_color="transparent", corner_radius=0,
                                                   scrollbar_button_color=PALETTE["bg"],
                                                   scrollbar_button_hover_color=PALETTE["fill2"])
        self.history_list.grid(row=1, column=0, sticky="nsew", padx=(34, 30), pady=(0, 22))

    @staticmethod
    def _pretty_date(iso):
        try:
            dt = datetime.fromisoformat(iso)
        except (TypeError, ValueError):
            return iso or ""
        return f"{dt.day} {MONTHS[dt.month - 1]} {dt.year}, {dt:%H:%M}"

    def _read_history(self):
        try:
            return json.loads(self.history_path.read_text(encoding="utf-8")) if self.history_path.exists() else []
        except (OSError, json.JSONDecodeError):
            return []

    def _write_history(self, entries):
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.history_path.with_suffix(".tmp")
        temp.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.history_path)

    def refresh_history(self):
        if not hasattr(self, "history_list"):
            return
        for w in self.history_list.winfo_children():
            w.destroy()
        entries = self._read_history()
        self._set_badge("history", str(len(entries)) if entries else "")
        if not entries:
            empty = ctk.CTkFrame(self.history_list, fg_color="transparent")
            empty.pack(fill="x", pady=80)
            self._label(empty, ICON["history"], ctk.CTkFont(family="Segoe MDL2 Assets", size=40), "text3").pack()
            self._label(empty, "История пуста", self.f_title3).pack(pady=(14, 2))
            self._label(empty, "Каждый расчёт сохраняется здесь автоматически", self.f_body, "text2").pack()
            return
        for idx, entry in enumerate(entries[:80]):
            res = entry.get("summary", {})
            ok = bool(res.get("ok"))
            row = self._card(self.history_list, radius=14)
            row.pack(fill="x", padx=6, pady=5)
            self._label(row, "●", self.f_dot, "green" if ok else "red", width=18).pack(side="left", padx=(18, 8))
            info = ctk.CTkFrame(row, fg_color="transparent")
            info.pack(side="left", fill="x", expand=True, pady=12)
            inputs = entry.get("inputs", {})
            title = entry.get("name", "Расчёт")
            if inputs:
                title += f"  ·  {inputs.get('b_mm', '?'):g}×{inputs.get('h_mm', '?'):g} мм, L = {inputs.get('L_m', '?'):g} м"
            self._label(info, title, self.f_body_sb, anchor="w").pack(fill="x")
            self._label(info, self._pretty_date(entry.get("date", "")) + "  ·  " +
                        ("успешно" if ok else "не сошёлся / нет результата"),
                        self.f_caption, "text2", anchor="w").pack(fill="x")
            ctk.CTkButton(row, text=ICON["delete"], width=34, height=34, corner_radius=17,
                          font=self.f_icon_s, fg_color="transparent", hover_color=tint("red", 0.14),
                          text_color=PALETTE["red"],
                          command=lambda i=idx: self.delete_history_entry(i)).pack(side="right", padx=(4, 14))
            self._pill(row, "Открыть", lambda i=idx: self.load_history_entry(i), width=96).pack(side="right", padx=4)
            for text in (f"M = {res.get('moment')} кН·м" if res.get("moment") not in (None, "—") else None,
                         res.get("reinforcement") if res.get("reinforcement") not in (None, "—") else None,
                         f"{res.get('iterations', 0)} {plural(res.get('iterations', 0) or 0, ('итерация', 'итерации', 'итераций'))}"):
                if text:
                    ctk.CTkLabel(row, text=text, font=self.f_caption_sb, fg_color=PALETTE["fill"],
                                 text_color=PALETTE["text"], corner_radius=8, height=26).pack(side="right", padx=4, ipadx=6)

    def _store_history(self, data, result, name="Расчёт"):
        entries = self._read_history()
        ok = bool(result and result.get("ok"))
        summary = {"ok": ok,
                   "iterations": len(result.get("iterations", [])) if result else 0,
                   "reinforcement": (f"{int(data.get('n'))}Ø{result.get('diameter_mm', 0):g}" if ok else "—"),
                   "moment": (f"{result.get('M', 0)/1000:.3f}" if ok else "—")}
        entries.insert(0, {"date": datetime.now().astimezone().isoformat(timespec="seconds"),
                           "name": name, "inputs": data, "summary": summary})
        try:
            self._write_history(entries[:200])
            self.refresh_history()
        except OSError as exc:
            messagebox.showwarning("История недоступна", f"Расчёт выполнен, но запись истории не удалась: {exc}")

    def load_history_entry(self, idx):
        try:
            entry = self._read_history()[idx]
            self._set_inputs(entry["inputs"])
            self.clear()
            self.show_page("calc")
            self.show_toast("Исходные данные загружены", ICON["open"])
        except (IndexError, KeyError, ValueError, TypeError) as exc:
            messagebox.showerror("История", f"Не удалось загрузить запись: {exc}")

    def delete_history_entry(self, idx):
        if not messagebox.askyesno("Удаление записи", "Удалить выбранную запись из истории? "
                                   "Исходный JSON-файл, если он сохранён отдельно, не удаляется."):
            return
        entries = self._read_history()
        if 0 <= idx < len(entries):
            del entries[idx]
            self._write_history(entries)
            self.refresh_history()

    # ------------------------------------------------------------------
    # Всплывающее уведомление
    # ------------------------------------------------------------------

    def _build_toast(self):
        self.toast = ctk.CTkFrame(self, corner_radius=22, fg_color=("#1D1D1F", "#F5F5F7"), height=44)
        self.toast_icon = ctk.CTkLabel(self.toast, text="", font=self.f_icon_s, text_color=("#30D158", "#248A3D"))
        self.toast_icon.pack(side="left", padx=(18, 8), pady=10)
        self.toast_label = ctk.CTkLabel(self.toast, text="", font=self.f_body_sb, text_color=("#FFFFFF", "#1D1D1F"))
        self.toast_label.pack(side="left", padx=(0, 20), pady=10)

    def show_toast(self, text, glyph=ICON["check"]):
        self.toast_icon.configure(text=glyph)
        self.toast_label.configure(text=text)
        self._cancel("toast_hide")
        self.toast.lift()

        def place(y):
            self.toast.place(in_=self.content, relx=0.5, rely=1.0, anchor="s", y=y)

        self._animate("toast", 380, lambda e: place(70 - 100 * e))

        def hide():
            def frame(e):
                place(-30 + 100 * e)
                if e >= 1:
                    self.toast.place_forget()
            self._animate("toast", 320, frame)
        self._jobs["toast_hide"] = self.after(2400, hide)

    # ------------------------------------------------------------------
    # Тема
    # ------------------------------------------------------------------

    def _set_appearance(self, value):
        ctk.set_appearance_mode("dark" if value == "Тёмное" else "light")
        for w in self._mpl_widgets:
            w.configure(bg=self.col("card"))
        self._config_protocol_tags()
        self.res_widget.configure(bg=self.col("bg"))
        self._draw_result_card(1.0)
        self._draw_bars()
        self._draw_hero()
        self._stale.update({"diagrams", "section", "iter"})
        self._redraw(self.current_page) if self.current_page in self._stale else None

    # ------------------------------------------------------------------
    # Проекты
    # ------------------------------------------------------------------

    def save_project_as(self):
        try:
            data = self._current_inputs()
        except ValueError as exc:
            self._set_banner("error", "Не удалось сохранить", str(exc))
            return
        path = filedialog.asksaveasfilename(title="Сохранить проект", defaultextension=".json",
                                            filetypes=(("Проект расчёта (*.json)", "*.json"), ("Все файлы", "*.*")))
        if not path:
            return
        payload = {"format": "crack_beam_project", "version": 1,
                   "saved_at": datetime.now().astimezone().isoformat(), "inputs": data}
        try:
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._store_history(data, self.result, Path(path).stem)
            self.show_toast(f"Проект «{Path(path).stem}» сохранён", ICON["save"])
        except OSError as exc:
            messagebox.showerror("Ошибка сохранения", str(exc))

    def open_project(self):
        path = filedialog.askopenfilename(title="Открыть проект",
                                          filetypes=(("Проект расчёта (*.json)", "*.json"), ("Все файлы", "*.*")))
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            data = payload.get("inputs", payload)
            self._set_inputs(data)
            self.clear()
            self.show_page("calc")
            self.show_toast(f"Открыт проект «{Path(path).stem}»", ICON["open"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
            messagebox.showerror("Не удалось открыть проект", f"Файл не похож на сохранённый проект: {exc}")

    # ------------------------------------------------------------------
    # Расчёт и вывод результатов
    # ------------------------------------------------------------------

    def clear(self):
        self.result = None
        self.result_inputs = None
        self._reset_outputs()
        self._set_banner("idle")
        self._render_protocol("Протокол пуст\n\nЗдесь появится подробный расчёт после нажатия кнопки «Рассчитать».")
        self._set_iterations([])
        self._refresh_previews()

    def run(self):
        try:
            data = self._current_inputs()
        except ValueError as e:
            self._set_banner("error", "Проверьте исходные данные", str(e))
            self.show_page("calc")
            return
        try:
            result = calculate(data)
        except ValueError as e:
            self.result = None
            self.result_inputs = None
            self._reset_outputs()
            partial = getattr(e, "iterations", None)
            self._set_banner("error", "Расчёт остановлен",
                             f"{e} Непроверенный результат не выведен как успешный.")
            if partial:
                self._render_protocol("РАСЧЁТ ОСТАНОВЛЕН — ЖУРНАЛ ХОДА РЕШЕНИЯ\n\n" + str(e) + "\n\n" +
                                      "".join(self.format_iteration(z) for z in partial))
                self._set_iterations(partial)
                self._store_history(data, {"ok": False, "iterations": partial}, "Остановленный расчёт")
            else:
                self._set_iterations([])
            self.show_page("calc")
            self._refresh_previews()
            return
        except Exception as e:
            messagebox.showerror(
                "Непредвиденная ошибка программы",
                f"Возникла ошибка типа {type(e).__name__}: {e}\n\n"
                "Сохраните исходные данные и передайте это сообщение разработчику. "
                "Результат расчёта не следует использовать до выяснения причины."
            )
            return
        self.result = result
        self.result_inputs = dict(data) if result.get("ok") else None
        self.show_result(result)
        self._store_history(data, result, "Расчёт")

    @staticmethod
    def fmt(x, digits=6):
        if x is None:
            return "-"
        return f"{x:.{digits}g}"

    def show_result(self, r):
        steps = r.get("iterations", [])
        self._set_iterations(steps)
        if not r.get("ok", False):
            self._reset_outputs()
            self._set_banner("error", "Расчёт не завершён", r.get("message", "Неизвестная ошибка"))
            text = "РАСЧЕТ НЕ ВЫПОЛНЕН\n\n" + r.get("message", "Неизвестная ошибка")
            if steps:
                text += "\n\nПОДРОБНЫЙ ХОД ИТЕРАЦИЙ, ВЫПОЛНЕННЫХ ДО ОСТАНОВКИ:\n" + "".join(
                    self.format_iteration(z) for z in steps)
            self._render_protocol(text)
            self._refresh_previews()
            self.show_toast("Расчёт не сошёлся", ICON["warning"])
            return

        n = int(r["inputs"]["n"])
        conc, steel = CONCRETES[r["inputs"]["concrete"]], STEELS[r["inputs"]["steel"]]
        self._set_banner("ok", draw=False)
        self._animate("res", 1000, self._draw_result_card)

        s_s, s_b = r["sigma_s"] / 1e6, r["sigma_b"] / 1e6
        Rs, Rb = steel.Rs / 1e6, conc.Rb / 1e6
        self._set_tile("rebar", f"{n} × Ø{r['diameter_mm']:.0f}", "",
                       f"As = {r['As']*1e6:.1f} мм² · {r['inputs']['steel']}")
        self._set_tile("mcrc", f"{r['Mcrc']/1000:.3f}", "кН·м",
                       f"M / Mcrc = {r['M']/r['Mcrc']:.2f} · Mb = {r['Mb']/1000:.3f} кН·м")
        self._set_tile("crack", f"{r['S_h0'] * r['h0'] * 1000:.0f}", "мм",
                       f"S / h₀ = {r['S_h0']:.3f} · h₀ = {r['h0']*1000:.0f} мм", ratio=r["S_h0"])
        self._set_tile("sigma_s", f"{s_s:.0f}", "МПа", f"{s_s / Rs * 100:.0f} % от Rs = {Rs:.0f} МПа", ratio=s_s / Rs)
        self._set_tile("sigma_b", f"{s_b:.2f}", "МПа", f"{s_b / Rb * 100:.0f} % от Rb = {Rb:.1f} МПа", ratio=s_b / Rb)
        self._set_tile("graph", f"{r['K']:.3f}", "K",
                       f"ξ₁ = {r['xi1']:.3f} · α₀m = {r['alpha0m']:.4f} · Lp = {r['Lp']:.2f}")
        self._draw_bars()

        self._render_protocol(self._protocol_text(r))
        self._refresh_previews()
        self.show_toast(f"Готово · M = {r['M']/1000:.3f} кН·м", ICON["check"])

    def _protocol_text(self, r):
        c = CONCRETES[r["inputs"]["concrete"]]
        s = STEELS[r["inputs"]["steel"]]
        parts = [
            "РАСЧЕТ МАЛОАРМИРОВАННОЙ ЖБ БАЛКИ\n",
            "=" * 72 + "\n\n",
            "1. ИСХОДНЫЕ ДАННЫЕ\n",
            f"b = {r['b']*1000:.1f} мм\n"
            f"h = {r['h']*1000:.1f} мм\n"
            f"a = {r['a']*1000:.1f} мм\n"
            f"h0 = {r['h0']*1000:.1f} мм\n"
            f"L = {r['L']:.3f} м\n"
            f"X = {r['X']*1000:.1f} мм\n"
            f"Бетон = {r['inputs']['concrete']}\n"
            f"Арматура = {r['inputs']['steel']}\n"
            f"n = {int(r['inputs']['n'])} шт.\n\n",
            "2. МАТЕРИАЛЫ\n",
            f"Eb = {c.Eb/1e9:.3f} ГПа\n"
            f"Rb = {c.Rb/1e6:.3f} МПа\n"
            f"Rbt = {c.Rbt/1e6:.3f} МПа\n"
            f"Rb,ser = {c.Rb_ser/1e6:.3f} МПа\n"
            f"ψb = {c.psi_b:.3f}\n"
            f"Es = {s.Es/1e9:.3f} ГПа\n"
            f"Rs = {s.Rs/1e6:.3f} МПа\n"
            f"Rsc = {s.Rsc/1e6:.3f} МПа\n"
            f"Ea (расчётное сопротивление арматуры) = {s.Ea/1e6:.3f} МПа\n"
            f"ψs = {r['psi_s']:.3f}\n\n",
            "3. ГЕОМЕТРИЧЕСКИЕ ХАРАКТЕРИСТИКИ\n",
            f"A = {r['A']:.6e} м²\n"
            f"I = {r['I']:.6e} м⁴\n"
            f"W = {r['W']:.6e} м³\n"
            f"h0 = {r['h0']:.6f} м\n\n",
            "4. НАГРУЗКА И ПЕРВИЧНЫЙ РАСЧЕТ\n",
            f"Собственный вес = {r['q_self']/1000:.4f} кН/м\n"
            f"Полная нагрузка q = {r['q_total']/1000:.4f} кН/м\n"
            f"Ra = {r['Ra']/1000:.4f} кН\n"
            f"Rb = {r['Rb']/1000:.4f} кН\n"
            f"M0(X) = {r['M0']/1000:.6f} кН·м\n"
            f"Q0(X) = {r['Q0']/1000:.6f} кН\n"
            f"N0 = {r['N0']/1000:.6f} кН\n\n",
            "5. ПОДБОР АРМАТУРЫ\n",
            f"α0 = Es/Eb = {r['alpha0']:.6f}\n"
            f"ξ = {r['xi']:.8f}\n"
            f"As0 = {r['As0']*1e6:.3f} мм² = {r['As0']*1e4:.3f} см²\n"
            f"Принято: {int(r['inputs']['n'])} стержней Ø{r['diameter_mm']:.0f} мм\n"
            f"As = {r['As']*1e6:.3f} мм² = {r['As']*1e4:.3f} см²\n"
            f"α0m = {r['alpha0m']:.6f}\n\n",
            "6. Mcrc\n",
            f"yt = {r['yt']*1000:.4f} мм\n"
            f"Wpl = {r['Wpl']:.6e} м³\n"
            f"Mcrc = {r['Mcrc']/1000:.6f} кН·м\n\n",
            "7. ГРАФИКИ ПЕРЕСЫПКИНА\n",
            f"L = {r['Lp']:.6f}\n"
            f"S/h0 = {r['S_h0']:.6f}\n"
            f"K = Mb/Mcrc = {r['K']:.6f}\n"
            f"ξ1 = {r['xi1']:.6f}\n"
            f"Mb = {r['Mb']/1000:.6f} кН·м\n\n",
            "8. НАПРЯЖЕНИЯ\n",
            f"σs = {r['sigma_s']/1e6:.6f} МПа\n"
            f"σb = {r['sigma_b']/1e6:.6f} МПа\n\n",
            "9. КАНОНИЧЕСКИЕ УРАВНЕНИЯ И ЕДИНИЦЫ\n",
            f"δMM = {r['deltaMM']:.6e}  1/(Н·м)\n"
            f"δMQ = δQM = {r['deltaMQ']:.6e}  1/Н\n"
            f"δQQ = {r['deltaQQ']:.6e}  м/Н\n"
            f"ΔMP = {r['deltaMP']:.6e}  безразмерная величина\n"
            f"ΔQP = {r['deltaQP']:.6e}  м\n"
            f"δM* = {r['deltaM_star']:.6e}  1/(Н·м) [по формуле задания]\n\n",
            "10. РАЗМЕРНОСТИ ОСНОВНЫХ ВЕЛИЧИН\n",
            "b, h, a, h0, X, d, yt — м\n"
            "A — м²; I — м⁴; W, Wpl — м³\n"
            "q — Н/м; Ra, Rb, Q, N — Н; M, M0, Mb, Mcrc — Н·м\n"
            "Eb, Es, Rb, Rbt, Rb,ser, σs, σb — Па; Ea — МПа (сопротивление арматуры)\n"
            "As, As0 — м²; α0, α0m, ξ, ξ1, K, ψb, ψs — безразмерные\n"
            "L (по формуле Пересыпкина), S/h0 — безразмерные\n\n",
            "11. ПОДРОБНЫЙ ХОД ИТЕРАЦИОННОГО РАСЧЁТА\n",
            "Каждый блок ниже — отдельный реально выполненный шаг. "
            "M0/Q0 — значения на входе шага; M/Q — решение канонической системы, "
            "которое становится входом следующего шага, если сходимость ещё не достигнута.\n\n",
        ]
        parts.extend(self.format_iteration(z) for z in r["iterations"])
        final_error = r["iterations"][-1]["error"]
        parts += [
            "\n" + "=" * 72 + "\n",
            "ИТОГ\n",
            f"Окончательный M = {r['M']/1000:.6f} кН·м\n"
            f"Окончательный Q = {r['Q']/1000:.6f} кН\n"
            f"Количество итераций = {len(r['iterations'])}\n"
            f"Итоговая относительная разница = {final_error:.3f}%\n"
            f"Арматура: {int(r['inputs']['n'])}Ø{r['diameter_mm']:.0f} "
            f"(As = {r['As']*1e4:.3f} см²)\n",
        ]
        return "".join(parts)

    def format_iteration(self, z):
        """Текстовый отчёт по одному полному итерационному шагу."""
        lines = [
            "-" * 72,
            f"ИТЕРАЦИЯ № {z['i']}",
            f"Вход шага: M0 = {z['M0']/1000:.8f} кН·м; Q0 = {z['Q0']/1000:.8f} кН",
            "1) Параметры сечения и арматуры:",
            "   Формулы: h0 = h − a; I = b·h³/12; ξ = M0/(Rb·b·h0²).",
            "   As0 = ξ·b·h0·(Rb/Ea) — требуемая площадь арматуры.",
            f"   Результат: ξ = {z['xi']:.8g}; As0 = {z['As0']*1e6:.6f} мм²",
            "   Подбор: выбирается минимальный табличный диаметр, для которого As = n·Aстержня ≥ As0.",
            f"   Принято n стержней Ø{z['diameter_mm']:.0f} мм; As = {z['As']*1e6:.6f} мм²",
            "   α0 = Es/Eb; α0m = (As/(b·h0))·α0.",
            f"   Результат: α0 = {z['alpha0']:.8g}; α0m = {z['alpha0m']:.8g}",
            "   Центр тяжести приведённого сечения: yt = [b·h²/2 + As·α0·a]/[b·h + As].",
            f"   Результат: yt = {z['yt']*1000:.6f} мм; Wpl = {z['Wpl']:.8e} м³",
            "   Wpl рассчитывается по формуле приведённого сечения с коэффициентом 1,3.",
            "   Mcrc = Rbt·Wpl + N0·[0,76923·Wpl/(b·h + As)].",
            f"   Результат: Mcrc = {z['Mcrc']/1000:.8f} кН·м",
        ]
        if z.get("stage_error"):
            lines.extend([
                "РАСЧЁТ ОСТАНОВЛЕН НА ЭТАПЕ ПОЛУЧЕНИЯ ПАРАМЕТРОВ ГРАФИКА:",
                "   " + z["stage_error"],
                "   Последующие величины на этом шаге не вычислялись.",
                "\n",
            ])
            return "\n".join(lines) + "\n"
        lines.extend([
            "2) Параметры по графикам Пересыпкина:",
            "   Lp = 1,7·10⁶·b·h0^1,5/M0; по Lp и α0m на графике определяется S/h0.",
            f"   Результат: Lp = {z['Lp']:.8g}; S/h0 = {z['S_h0']:.8g}; S = (S/h0)·h0 = {z['S']*1000:.6f} мм",
            "   K и ξ1 считываются/интерполируются по графикам при α0m и S/h0; Mb = K·Mcrc.",
            f"   Результат: K = {z['K']:.8g}; ξ1 = {z['xi1']:.8g}; Mb = {z['Mb']/1000:.8f} кН·м",
            "3) Напряжения и податливость трещины:",
            "   σs = (M0 − Mb)/[As0·(0,98·h0)].",
            f"   Результат: σs = {z['sigma_s']/1e6:.8f} МПа",
            "   σb = [2·σs·As0 + Mb/(0,6667·(h − S))]/(ξ1·b·h0).",
            f"   Результат: σb = {z['sigma_b']/1e6:.8f} МПа",
            "   δM* = 2h/(M0·h0)·(σs·ψs/Es + σb·ψb/Eb)² − 2h/[Eb·(Wpl·yt/1,3)].",
            f"   Результат: δM* = {z['deltaM_star']:.8e}",
            "4) Коэффициенты канонической системы:",
            "   Коэффициенты получены интегрированием произведений эпюр: δij = ∫Mi·Mj/(E·I) dx; ΔiP = ∫Mi·Mнаг/(E·I) dx.",
            f"   δMM = {z['deltaMM']:.8e}; δMQ = δQM = {z['deltaMQ']:.8e}; δQQ = {z['deltaQQ']:.8e}",
            f"   ΔMP = {z['deltaMP']:.8e}; ΔQP = {z['deltaQP']:.8e}",
            "5) Результат решения системы:",
            "   [δMM + δM*   δMQ]·[M] = [−ΔMP];  [δMQ   δQQ]·[Q]   [−ΔQP].",
            f"   M = {z['M']/1000:.8f} кН·м; Q = {z['Q']/1000:.8f} кН",
            f"   относительное изменение момента = {z['error']:.6f}%",
            "   проверка момента: " + (f"выполнена (≤ {MOMENT_TOLERANCE_PERCENT:g}%)" if z['error'] <= MOMENT_TOLERANCE_PERCENT else f"не выполнена (> {MOMENT_TOLERANCE_PERCENT:g}%)"),
            f"   диаметр для следующего состояния: Ø{z.get('diameter_next_mm', z['diameter_mm']):.0f} мм; " + ("подбор стабилен" if z.get("diameter_stable", True) else "диаметр изменится — нужна следующая итерация"),
            "   итог шага: " + ("сходимость подтверждена" if z.get("converged_step", z["error"] <= MOMENT_TOLERANCE_PERCENT) else "переход к следующей итерации"),
            "\n",
        ])
        return "\n".join(lines) + "\n"


if __name__ == "__main__":
    App().mainloop()
