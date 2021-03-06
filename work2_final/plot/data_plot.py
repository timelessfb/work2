#!usr/bin/env python
# -*- coding:utf-8 -*-
import datetime
import sys
import time
import numpy as np
import random
import work2_final.plot.plot as pt

cost_result = np.array([[[0.57142584, 0.54770766, 0.55844131, 1.67757481],
                         [1.14923997, 1.15368282, 1.17591458, 3.47883737],
                         [1.83486556, 1.8671315, 1.84859931, 5.55059637],
                         [2.57721353, 2.63827915, 2.65301725, 7.86850993],
                         [3.53896094, 3.57237513, 3.57492771, 10.68626377],
                         [4.6267443, 4.61170969, 4.60434847, 13.84280246],
                         [6.05155644, 6.09680675, 6.05660305, 18.20496625],
                         [7.32271566, 7.34097908, 7.35691747, 22.02061221],
                         [7.47860346, 7.46259537, 7.4978087, 22.43900752],
                         [7.49896788, 7.45024695, 7.54687785, 22.49609268]],

                        [[0.56977611, 0.54920493, 0.56692509, 1.68590613],
                         [1.16513756, 1.15890446, 1.18640836, 3.51045038],
                         [1.8491652, 1.86878483, 1.87854708, 5.59649711],
                         [2.59143557, 2.65129545, 2.69114544, 7.93387645],
                         [3.54546105, 3.6010636, 3.61077365, 10.7572983],
                         [4.66285538, 4.64804383, 4.63247237, 13.94337158],
                         [5.9267549, 6.01003559, 5.90045947, 17.83724996],
                         [6.89186739, 6.9331661, 6.97403352, 20.79906701],
                         [7.14014868, 7.22616512, 7.23203885, 21.59835266],
                         [7.29355828, 7.38310594, 7.40206752, 22.07873174]],

                        [[0.51700299, 0.49438595, 1.01902068, 2.03040961],
                         [1.0990747, 1.09214301, 2.00776977, 4.19898748],
                         [1.79524622, 1.80015216, 2.83978874, 6.43518712],
                         [2.54515791, 2.61532557, 3.86016489, 9.02064837],
                         [3.54383512, 3.54646079, 4.80076805, 11.89106397],
                         [4.70961702, 4.69765987, 5.99434389, 15.40162079],
                         [5.7339906, 5.80548974, 7.07254388, 18.61202423],
                         [6.42357112, 6.51844347, 7.71523493, 20.65724952],
                         [6.73185852, 6.82782974, 7.97881587, 21.53850413],
                         [6.92399542, 7.06339423, 8.20060502, 22.18799467]],

                        [[0.91415678, 0.42633021, 0.90726355, 2.24775055],
                         [1.84831244, 0.98953019, 2.00822415, 4.84606677],
                         [2.94560153, 1.67655405, 3.00148875, 7.62364433],
                         [3.86417272, 2.5648936, 3.9961867, 10.42525303],
                         [4.8972956, 3.64936588, 4.9698821, 13.51654358],
                         [5.92125075, 4.71006852, 6.00099372, 16.632313],
                         [6.68547717, 5.55573454, 6.76076601, 19.00197772],
                         [7.17509051, 6.10460135, 7.28399519, 20.56368706],
                         [7.44914823, 6.36742196, 7.50964991, 21.3262201],
                         [7.65050668, 6.58229221, 7.71209158, 21.94489047]],

                        [[0.89987822, 0.87325857, 0.44960021, 2.222737],
                         [1.85027021, 1.97306708, 1.03006713, 4.85340442],
                         [2.9484738, 3.00388414, 1.69251034, 7.64486828],
                         [3.94794319, 4.03216618, 2.57657248, 10.55668184],
                         [4.87438799, 4.95940285, 3.63227473, 13.46606557],
                         [5.90021439, 5.93195451, 4.71457801, 16.54674692],
                         [6.66840551, 6.70283965, 5.62830565, 18.99955081],
                         [7.11020193, 7.15805216, 6.11562504, 20.38387912],
                         [7.42736065, 7.41331408, 6.36213627, 21.202811],
                         [7.64087495, 7.58178816, 6.55765481, 21.78031792]],

                        [[0.56576932, 0.55475335, 0.59186612, 1.71238879],
                         [1.16700417, 1.16979227, 1.22986075, 3.56665718],
                         [1.87383874, 1.88182601, 1.8793713, 5.63503605],
                         [2.63106184, 2.66911355, 2.69532248, 7.99549787],
                         [3.59530234, 3.59163536, 3.58093369, 10.76787139],
                         [4.68971934, 4.6495203, 4.60878045, 13.94802009],
                         [5.89361774, 5.97204634, 5.83507738, 17.70074146],
                         [6.8908494, 6.89948449, 6.91827234, 20.70860623],
                         [7.27090297, 7.29874395, 7.23444352, 21.80409044],
                         [7.44443569, 7.45655802, 7.43006012, 22.33105383]],

                        [[0.97439994, 0.83995011, 0.91030682, 2.72465687],
                         [1.96297002, 1.82685595, 1.86708554, 5.6569115],
                         [3.05051326, 2.91415675, 2.87398775, 8.83865776],
                         [3.95241253, 4.03085453, 3.9896436, 11.97291067],
                         [4.95760501, 4.99154258, 5.05534284, 15.00449044],
                         [5.79441448, 5.79647077, 5.87193653, 17.46282179],
                         [6.31621906, 6.29897484, 6.38061539, 18.99580929],
                         [6.61240224, 6.60455711, 6.68160643, 19.89856579],
                         [6.88697335, 6.91329111, 6.95679198, 20.75705643],
                         [7.08898929, 7.12908004, 7.16906468, 21.38713401]]])
fails = np.array([[0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
                   6.900e-01, 3.250e+00, 6.130e+00],
                  [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 1.000e-02, 5.000e-02, 2.900e-01,
                   1.330e+00, 3.820e+00, 6.500e+00],
                  [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 8.000e-02, 7.900e-01,
                   2.410e+00, 4.840e+00, 7.420e+00],
                  [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 1.000e-02, 3.600e-01, 1.560e+00,
                   3.460e+00, 5.950e+00, 8.530e+00],
                  [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 1.000e-02, 3.700e-01, 1.500e+00,
                   3.540e+00, 5.980e+00, 8.590e+00],
                  [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 1.000e-02, 8.000e-02, 3.500e-01,
                   1.340e+00, 3.630e+00, 6.290e+00],
                  [0.000e+00, 0.000e+00, 0.000e+00, 2.000e-02, 1.700e-01, 1.060e+00, 2.940e+00,
                   5.280e+00, 7.720e+00, 1.029e+01]])
req_num_eachtime = 3
nowtime = (lambda: int(round(time.time() * 1000)))
nowtime = nowtime()
max_iter = 200
n = 100
pt.plot_fun_slot(cost_result[:, :, 2], fails, req_num_eachtime, '切片请求数量（个）', 1,
                 str(nowtime) + '计算资源映射代价')
pt.plot_fun_slot((cost_result[:, :, 0] + cost_result[:, :, 1]), fails, req_num_eachtime, '切片请求数量（个）',
                 2,
                 str(nowtime) + '带宽资源映射代价')
pt.plot_fun_slot(cost_result[:, :, 3], fails, req_num_eachtime, '切片请求数量（个）', 3,
                 str(nowtime) + '总映射代价' + '_' + str(max_iter) + '_' + str(n))
pt.plot_fun_fail_slot(fails, req_num_eachtime, '切片请求数量（个）', '失败率', str(nowtime) + '失败率')
