import cv2
import numpy as np
import random
import math

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih

    areaA = (ax2 - ax1) * (ay2 - ay1)
    areaB = (bx2 - bx1) * (by2 - by1)
    union = areaA + areaB - inter

    return inter / union if union else 0.0

def hungarian_stars(cost_matrix):
    cost = np.array(cost_matrix, dtype=float)
    n, m = cost.shape

    for i in range(n):
        cost[i] -= np.min(cost[i])

    for j in range(n):
        cost[:, j] -= np.min(cost[:, j])

    stars = np.zeros((n, n), dtype=bool)
    primes = np.zeros((n, n), dtype=bool)

    covered_rows = np.zeros(n, dtype=bool)
    covered_cols = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if np.isclose(cost[i, j], 0) and not covered_rows[i] and not covered_cols[j]:
                stars[i, j] = True
                covered_rows[i] = True
                covered_cols[j] = True

    covered_rows[:] = False
    covered_cols[:] = False

    def cover_columns_with_stars():
        for j in range(n):
            if np.any(stars[:, j]):
                covered_cols[j] = True

    cover_columns_with_stars()

    while True:
        if covered_cols.sum() == n:
            break

        while True:
            z = np.where((np.isclose(cost, 0)) & (~covered_rows[:, None]) & (~covered_cols[None, :]))
            if len(z[0]) == 0:
                uncovered = cost[~covered_rows][:, ~covered_cols]
                min_uncovered = np.min(uncovered)

                cost[~covered_rows] -= min_uncovered
                cost[:, covered_cols] += min_uncovered
                continue

            row, col = z[0][0], z[1][0]
            primes[row, col] = True

            star_in_row = np.where(stars[row])[0]
            if len(star_in_row) == 0:
                path = [(row, col)]

                while True:
                    r = np.where(stars[:, col])[0]
                    if len(r) == 0:
                        break
                    r = r[0]
                    path.append((r, col))

                    c = np.where(primes[r])[0]
                    c = c[0]
                    path.append((r, c))
                    col = c

                for r, c in path:
                    stars[r, c] = not stars[r, c]
                    primes[r, c] = False

                primes[:] = False
                covered_rows[:] = False
                covered_cols[:] = False
                cover_columns_with_stars()
                break
            else:
                covered_rows[row] = True
                covered_cols[star_in_row[0]] = False

    result = np.where(stars)
    matches = list(zip(result[0], result[1]))
    return matches