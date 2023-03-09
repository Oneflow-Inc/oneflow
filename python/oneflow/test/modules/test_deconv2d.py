"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest
import torch as torch_original
from packaging import version


def _test_deconv_bias_false(test_case, device):
    np_arr = np.array(
        [
            [
                [
                    [0.2735021114349365, -1.3842310905456543],
                    [1.058540940284729, -0.03388553857803345],
                ]
            ]
        ]
    )
    input = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    weight = np.array(
        [
            [
                [
                    [0.06456436216831207, -0.10852358490228653, -0.21638715267181396],
                    [-0.2279110550880432, 0.1476770043373108, 0.19457484781742096],
                    [0.05026858672499657, 0.10818571597337723, 0.02056501805782318],
                ],
                [
                    [0.205095112323761, 0.1488947868347168, -0.2344113141298294],
                    [0.1684819906949997, -0.21986986696720123, 0.1082606166601181],
                    [-0.1528974026441574, 0.17120417952537537, 0.01954500749707222],
                ],
            ]
        ]
    )
    m = nn.ConvTranspose2d(1, 2, 3, stride=1, bias=False)
    m.weight = flow.nn.Parameter(flow.Tensor(weight))
    m = m.to(device)
    output = m(input)
    np_out = np.array(
        [
            [
                [
                    [
                        0.01765848882496357,
                        -0.1190534234046936,
                        0.09103937447071075,
                        0.2995298206806183,
                    ],
                    [
                        0.006009865552186966,
                        0.2388070970773697,
                        -0.37657976150512695,
                        -0.26200416684150696,
                    ],
                    [
                        -0.22750461101531982,
                        0.12405071407556534,
                        0.056831881403923035,
                        -0.035060010850429535,
                    ],
                    [
                        0.053211357444524765,
                        0.11281562596559525,
                        0.0181029811501503,
                        -0.0006968567031435668,
                    ],
                ],
                [
                    [
                        0.05609394609928131,
                        -0.24317599833011627,
                        -0.27021679282188416,
                        0.32447943091392517,
                    ],
                    [
                        0.26318174600601196,
                        -0.14269141852855682,
                        0.08078087121248245,
                        -0.14191456139087677,
                    ],
                    [
                        0.13652732968330383,
                        0.020019691437482834,
                        -0.10959184169769287,
                        -0.03072327747941017,
                    ],
                    [
                        -0.16184815764427185,
                        0.1864076405763626,
                        0.014887845143675804,
                        -0.0006622931105084717,
                    ],
                ],
            ]
        ]
    )
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-06, 1e-06))
    output = output.sum()
    output.backward()
    np_grad = [
        [
            [
                [0.24731683731079102, 0.24731683731079102],
                [0.24731683731079102, 0.24731683731079102],
            ]
        ]
    ]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-06, 1e-06))


def _test_deconv_bias_true(test_case, device):
    np_arr = np.array(
        [
            [
                [
                    [0.2735021114349365, -1.3842310905456543],
                    [1.058540940284729, -0.03388553857803345],
                ]
            ]
        ]
    )
    input = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    weight = np.array(
        [
            [
                [
                    [0.06456436216831207, -0.10852358490228653, -0.21638715267181396],
                    [-0.2279110550880432, 0.1476770043373108, 0.19457484781742096],
                    [0.05026858672499657, 0.10818571597337723, 0.02056501805782318],
                ],
                [
                    [0.205095112323761, 0.1488947868347168, -0.2344113141298294],
                    [0.1684819906949997, -0.21986986696720123, 0.1082606166601181],
                    [-0.1528974026441574, 0.17120417952537537, 0.01954500749707222],
                ],
            ]
        ]
    )
    bias = np.array([0.06456436216831207, -0.10852358490228653])
    m = nn.ConvTranspose2d(1, 2, 3, stride=1)
    m.weight = flow.nn.Parameter(flow.Tensor(weight))
    m.bias = flow.nn.Parameter(flow.Tensor(bias))
    m = m.to(device)
    output = m(input)
    np_out = [
        [
            [
                [
                    0.0822228491306305,
                    -0.05448906123638153,
                    0.15560373663902283,
                    0.36409419775009155,
                ],
                [
                    0.07057422399520874,
                    0.30337145924568176,
                    -0.3120154142379761,
                    -0.19743980467319489,
                ],
                [
                    -0.16294024884700775,
                    0.188615083694458,
                    0.12139624357223511,
                    0.029504351317882538,
                ],
                [
                    0.11777572333812714,
                    0.17737999558448792,
                    0.08266734331846237,
                    0.06386750191450119,
                ],
            ],
            [
                [
                    -0.05242963880300522,
                    -0.3516995906829834,
                    -0.3787403702735901,
                    0.21595585346221924,
                ],
                [
                    0.15465816855430603,
                    -0.25121501088142395,
                    -0.027742713689804077,
                    -0.2504381537437439,
                ],
                [
                    0.028003744781017303,
                    -0.088503897190094,
                    -0.2181154191493988,
                    -0.139246866106987,
                ],
                [
                    -0.2703717350959778,
                    0.07788405567407608,
                    -0.09363573789596558,
                    -0.10918587446212769,
                ],
            ],
        ]
    ]
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-06, 1e-06))
    output = output.sum()
    output.backward()
    np_grad = [
        [
            [
                [0.24731683731079102, 0.24731683731079102],
                [0.24731683731079102, 0.24731683731079102],
            ]
        ]
    ]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-06, 1e-06))


def _test_deconv_group_bias_false(test_case, device):
    np_arr = np.array(
        [
            [
                [
                    [-2.0125174206754517, 1.9917882689443576],
                    [0.13146748727936577, -0.5356457374181375],
                ],
                [
                    [1.020683505853394, 1.2900643048299678],
                    [-0.549010560600543, 0.8088391626901512],
                ],
            ]
        ]
    )
    input = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m = nn.ConvTranspose2d(2, 2, 3, stride=1, groups=2, bias=False)
    weight = np.array(
        [
            [
                [
                    [0.06456436216831207, -0.10852358490228653, -0.21638715267181396],
                    [-0.2279110550880432, 0.1476770043373108, 0.19457484781742096],
                    [0.05026858672499657, 0.10818571597337723, 0.02056501805782318],
                ]
            ],
            [
                [
                    [0.205095112323761, 0.1488947868347168, -0.2344113141298294],
                    [0.1684819906949997, -0.21986986696720123, 0.1082606166601181],
                    [-0.1528974026441574, 0.17120417952537537, 0.01954500749707222],
                ]
            ],
        ]
    )
    m.weight = flow.nn.Parameter(flow.Tensor(weight))
    m = m.to(device)
    output = m(input)
    np_out = np.array(
        [
            [
                [
                    [
                        -0.12993690371513367,
                        0.34700414538383484,
                        0.219326913356781,
                        -0.43099740147590637,
                    ],
                    [
                        0.4671630859375,
                        -0.8000040054321289,
                        -0.06776165962219238,
                        0.5034587383270264,
                    ],
                    [
                        -0.13112929463386536,
                        0.02389305830001831,
                        0.12057329714298248,
                        -0.06326202303171158,
                    ],
                    [
                        0.00660868501290679,
                        -0.012703249230980873,
                        -0.05524558573961258,
                        -0.011015564203262329,
                    ],
                ],
                [
                    [
                        0.20933720469474792,
                        0.4165603518486023,
                        -0.04717591404914856,
                        -0.3024056851863861,
                    ],
                    [
                        0.059367403388023376,
                        0.07707919180393219,
                        0.07597976922988892,
                        -0.049937888979911804,
                    ],
                    [
                        -0.24855825304985046,
                        0.2344835251569748,
                        0.003538096323609352,
                        0.11277973651885986,
                    ],
                    [
                        0.08394229412078857,
                        -0.21766230463981628,
                        0.12774622440338135,
                        0.015808766707777977,
                    ],
                ],
            ]
        ]
    )

    test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-06, 1e-06))
    output = output.sum()
    output.backward()
    np_grad = [
        [
            [
                [0.03301373869180679, 0.03301373869180679],
                [0.03301373869180679, 0.03301373869180679],
            ],
            [
                [0.21430310606956482, 0.21430310606956482],
                [0.21430310606956482, 0.21430310606956482],
            ],
        ]
    ]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-06, 1e-06))


def _test_deconv_group_bias_true(test_case, device):
    np_arr = np.array(
        [
            [
                [
                    [-2.0125174206754517, 1.9917882689443576],
                    [0.13146748727936577, -0.5356457374181375],
                ],
                [
                    [1.020683505853394, 1.2900643048299678],
                    [-0.549010560600543, 0.8088391626901512],
                ],
            ]
        ]
    )
    input = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m = nn.ConvTranspose2d(2, 2, 3, stride=1, groups=2)
    weight = np.array(
        [
            [
                [
                    [0.06456436216831207, -0.10852358490228653, -0.21638715267181396],
                    [-0.2279110550880432, 0.1476770043373108, 0.19457484781742096],
                    [0.05026858672499657, 0.10818571597337723, 0.02056501805782318],
                ]
            ],
            [
                [
                    [0.205095112323761, 0.1488947868347168, -0.2344113141298294],
                    [0.1684819906949997, -0.21986986696720123, 0.1082606166601181],
                    [-0.1528974026441574, 0.17120417952537537, 0.01954500749707222],
                ]
            ],
        ]
    )
    m.weight = flow.nn.Parameter(flow.Tensor(weight))
    bias = np.array([0.06456436216831207, -0.10852358490228653])
    m.bias = flow.nn.Parameter(flow.Tensor(bias))
    m = m.to(device)
    output = m(input)
    np_out = [
        [
            [
                [
                    -0.0653725415468216,
                    0.4115685224533081,
                    0.2838912606239319,
                    -0.3664330244064331,
                ],
                [
                    0.5317274332046509,
                    -0.735439658164978,
                    -0.00319729745388031,
                    0.5680230855941772,
                ],
                [
                    -0.06656493246555328,
                    0.08845742046833038,
                    0.18513765931129456,
                    0.0013023391366004944,
                ],
                [
                    0.0711730495095253,
                    0.05186111479997635,
                    0.009318776428699493,
                    0.053548797965049744,
                ],
            ],
            [
                [
                    0.1008136197924614,
                    0.30803677439689636,
                    -0.1556994915008545,
                    -0.41092926263809204,
                ],
                [
                    -0.04915618151426315,
                    -0.03144439309835434,
                    -0.032543815672397614,
                    -0.15846148133277893,
                ],
                [
                    -0.3570818305015564,
                    0.12595993280410767,
                    -0.10498549044132233,
                    0.004256151616573334,
                ],
                [
                    -0.024581290781497955,
                    -0.3261858820915222,
                    0.019222639501094818,
                    -0.0927148163318634,
                ],
            ],
        ]
    ]
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-06, 1e-06))
    output = output.sum()
    output.backward()
    np_grad = [
        [
            [
                [0.03301373869180679, 0.03301373869180679],
                [0.03301373869180679, 0.03301373869180679],
            ],
            [
                [0.21430310606956482, 0.21430310606956482],
                [0.21430310606956482, 0.21430310606956482],
            ],
        ]
    ]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-06, 1e-06))


def _test_deconv_group_large_out_channel(test_case, device):
    np_arr = np.array(
        [
            [
                [
                    [-2.0125174206754517, 1.9917882689443576],
                    [0.13146748727936577, -0.5356457374181375],
                ],
                [
                    [1.020683505853394, 1.2900643048299678],
                    [-0.549010560600543, 0.8088391626901512],
                ],
            ]
        ]
    )
    input = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m = nn.ConvTranspose2d(2, 6, 3, stride=1, groups=2, bias=False)
    weight = np.array(
        [
            [
                [
                    [0.05271657928824425, -0.08860913664102554, -0.17667937278747559],
                    [-0.18608860671520233, 0.12057777494192123, 0.1588696986436844],
                    [0.04104413092136383, 0.08833327144384384, 0.016791267320513725],
                ],
                [
                    [0.16745945811271667, 0.1215720921754837, -0.19139604270458221],
                    [0.13756497204303741, -0.17952299118041992, 0.08839442580938339],
                    [-0.12484020739793777, 0.13978762924671173, 0.015958432108163834],
                ],
                [
                    [-0.07709092646837234, -0.029757702723145485, -0.18154984712600708],
                    [-0.14461342990398407, 0.06567336618900299, 0.05665326863527298],
                    [0.04441174864768982, -0.04477253183722496, 0.191376194357872],
                ],
            ],
            [
                [
                    [0.1850736141204834, 0.07141514122486115, 0.05791180208325386],
                    [0.07253318279981613, -0.042754165828228, -0.14045141637325287],
                    [0.08525089919567108, 0.009758883155882359, -0.07303793728351593],
                ],
                [
                    [-0.005451973062008619, 0.1499139368534088, 0.16706342995166779],
                    [-0.05473465472459793, 0.02753184549510479, -0.06856250017881393],
                    [0.03629609942436218, -0.06238799914717674, -0.041715867817401886],
                ],
                [
                    [0.15021666884422302, -0.10501708835363388, 0.04741475358605385],
                    [-0.16011257469654083, 0.1280348002910614, 0.11050418764352798],
                    [-0.10031674802303314, 0.1449088454246521, -0.16990724205970764],
                ],
            ],
        ]
    )
    m.weight = flow.nn.Parameter(flow.Tensor(weight))
    m = m.to(device)
    output = m(input)
    np_out = np.array(
        [
            [
                [
                    [
                        -0.10609303414821625,
                        0.28332769870758057,
                        0.17907968163490295,
                        -0.3519079089164734,
                    ],
                    [
                        0.3814370930194855,
                        -0.653200626373291,
                        -0.055327147245407104,
                        0.41107234358787537,
                    ],
                    [
                        -0.10706663131713867,
                        0.019508585333824158,
                        0.09844768047332764,
                        -0.05165322124958038,
                    ],
                    [
                        0.005395968910306692,
                        -0.010372160002589226,
                        -0.04510783404111862,
                        -0.00899417046457529,
                    ],
                ],
                [
                    [
                        -0.3370150923728943,
                        0.08887782692909241,
                        0.6273337602615356,
                        -0.38122040033340454,
                    ],
                    [
                        -0.25483641028404236,
                        0.561577320098877,
                        -0.6257490515708923,
                        0.27858346700668335,
                    ],
                    [
                        0.26932841539382935,
                        -0.6272678375244141,
                        0.35409244894981384,
                        -0.015562277287244797,
                    ],
                    [
                        -0.01641242951154709,
                        0.08524765074253082,
                        -0.0727786272764206,
                        -0.008548066020011902,
                    ],
                ],
                [
                    [
                        0.15514683723449707,
                        -0.09366090595722198,
                        0.3061012029647827,
                        -0.3616088628768921,
                    ],
                    [
                        0.28090208768844604,
                        -0.38282686471939087,
                        0.008863434195518494,
                        0.21008771657943726,
                    ],
                    [
                        -0.10839138925075531,
                        0.2646597623825073,
                        -0.5020549297332764,
                        0.35083478689193726,
                    ],
                    [
                        0.005838701035827398,
                        -0.029675094410777092,
                        0.04914196580648422,
                        -0.10250984132289886,
                    ],
                ],
                [
                    [
                        0.18890158832073212,
                        0.3116491138935089,
                        0.15123975276947021,
                        0.074709951877594,
                    ],
                    [
                        -0.027573950588703156,
                        0.16042113304138184,
                        -0.17254289984703064,
                        -0.1343500316143036,
                    ],
                    [
                        0.047192707657814026,
                        0.20208004117012024,
                        -0.01943095773458481,
                        -0.20782624185085297,
                    ],
                    [
                        -0.04680364578962326,
                        0.06359653919935226,
                        0.04799196869134903,
                        -0.05907594412565231,
                    ],
                ],
                [
                    [
                        -0.005564738996326923,
                        0.1459812968969345,
                        0.3639175295829773,
                        0.21552257239818573,
                    ],
                    [
                        -0.05287356674671173,
                        -0.12922403216362,
                        -0.0049260929226875305,
                        0.04667740315198898,
                    ],
                    [
                        0.06709674000740051,
                        -0.0762409120798111,
                        -0.06315286457538605,
                        -0.10927218943834305,
                    ],
                    [
                        -0.019926942884922028,
                        0.06360937654972076,
                        -0.027559401467442513,
                        -0.03374142572283745,
                    ],
                ],
                [
                    [
                        0.1533236801624298,
                        0.08659995347261429,
                        -0.08708333969116211,
                        0.06116808205842972,
                    ],
                    [
                        -0.24589480459690094,
                        0.10328409075737,
                        0.16698980331420898,
                        0.1809084266424179,
                    ],
                    [
                        -0.014488153159618378,
                        -0.18130677938461304,
                        0.056411802768707275,
                        -0.1298111528158188,
                    ],
                    [
                        0.05507495626807213,
                        -0.1606965959072113,
                        0.21048882603645325,
                        -0.13742762804031372,
                    ],
                ],
            ]
        ]
    )
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-06, 1e-06))
    output = output.sum()
    output.backward()
    np_grad = [
        [
            [
                [0.0822635293006897, 0.0822635293006897],
                [0.0822635293006897, 0.0822635293006897],
            ],
            [
                [0.4193778932094574, 0.4193778932094574],
                [0.4193778932094574, 0.4193778932094574],
            ],
        ]
    ]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-06, 1e-06))


def _test_deconv_group_large_in_channel(test_case, device):
    np_arr = [
        [
            [
                [0.6393764315295867, 0.3890587560476374],
                [0.8467359871201484, 0.24046160407703143],
            ],
            [
                [0.23352071016856402, 0.6760713653927521],
                [0.061939453383917376, 0.13541973098624682],
            ],
            [
                [0.7524804920779914, 0.34366296030931365],
                [0.4961502482687954, 0.38175448164636205],
            ],
            [
                [0.01867975512238773, 0.12599156959160163],
                [0.2658608593205851, 0.6184459583178925],
            ],
        ]
    ]
    input = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m = nn.ConvTranspose2d(4, 2, 3, stride=1, groups=2, bias=False)
    weight = np.array(
        [
            [
                [
                    [0.09130779653787613, -0.15347552299499512, -0.30601766705513],
                    [-0.32231491804122925, 0.2088468372821808, 0.27517038583755493],
                    [0.07109051942825317, 0.1529977172613144, 0.02908332832157612],
                ]
            ],
            [
                [
                    [0.2900483012199402, 0.21056903898715973, -0.33150768280029297],
                    [0.23826952278614044, -0.31094294786453247, 0.15310363471508026],
                    [-0.21622958779335022, 0.24211928248405457, 0.0276408139616251],
                ]
            ],
            [
                [
                    [-0.13352541625499725, -0.051541853696107864, -0.3144535720348358],
                    [-0.2504778206348419, 0.11374961584806442, 0.09812634438276291],
                    [0.07692340761423111, -0.0775483027100563, 0.33147329092025757],
                ]
            ],
            [
                [
                    [0.3205569088459015, 0.12369465827941895, 0.1003061905503273],
                    [0.1256311535835266, -0.07405238598585129, -0.24326899647712708],
                    [0.14765889942646027, 0.016902882605791092, -0.12650541961193085],
                ]
            ],
        ]
    )
    m.weight = flow.nn.Parameter(flow.Tensor(weight))
    m = m.to(device)
    np_out = np.array(
        [
            [
                [
                    [
                        0.12611234188079834,
                        0.1826610565185547,
                        -0.19042569398880005,
                        -0.34318169951438904,
                    ],
                    [
                        -0.05516064167022705,
                        0.04093143343925476,
                        -0.2053149938583374,
                        0.0920882523059845,
                    ],
                    [
                        -0.2631978690624237,
                        0.14817529916763306,
                        0.4988565742969513,
                        0.11690345406532288,
                    ],
                    [
                        0.04680176079273224,
                        0.13235820829868317,
                        0.09591575711965561,
                        0.010736535303294659,
                    ],
                ],
                [
                    [
                        -0.09448734670877457,
                        -0.04197392612695694,
                        -0.2368750274181366,
                        -0.09542831033468246,
                    ],
                    [
                        -0.1671580672264099,
                        0.16854587197303772,
                        0.02652890235185623,
                        -0.05493755638599396,
                    ],
                    [
                        -0.030232630670070648,
                        0.0058259665966033936,
                        0.20417997241020203,
                        -0.015012085437774658,
                    ],
                    [
                        0.07742229104042053,
                        0.0867031067609787,
                        0.11167682707309723,
                        0.048304662108421326,
                    ],
                ],
            ]
        ]
    )
    output = m(input)
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-06, 1e-06))
    output = output.sum()
    output.backward()
    np_grad = [
        [
            [
                [0.046688467264175415, 0.046688467264175415],
                [0.046688467264175415, 0.046688467264175415],
            ],
            [
                [0.30307042598724365, 0.30307042598724365],
                [0.30307042598724365, 0.30307042598724365],
            ],
            [
                [-0.20727425813674927, -0.20727425813674927],
                [-0.20727425813674927, -0.20727425813674927],
            ],
            [
                [0.3909238576889038, 0.3909238576889038],
                [0.3909238576889038, 0.3909238576889038],
            ],
        ]
    ]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-06, 1e-06))


@flow.unittest.skip_unless_1n1d()
class TestDeconv2d(flow.unittest.TestCase):
    def test_deconv2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_deconv_bias_false,
            _test_deconv_bias_true,
            _test_deconv_group_bias_false,
            _test_deconv_group_bias_true,
            _test_deconv_group_large_out_channel,
            _test_deconv_group_large_in_channel,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_deconv2d_with_random_data(test_case):
        channels = random(1, 6)
        m = torch.nn.ConvTranspose2d(
            in_channels=channels,
            out_channels=random(1, 20),
            kernel_size=random(1, 4),
            stride=random() | nothing(),
            padding=random(1, 3).to(int) | nothing(),
            dilation=random(1, 5) | nothing(),
            groups=random(1, 5) | nothing(),
            padding_mode=constant("zeros") | nothing(),
            bias=random_bool(),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=4, dim1=channels).to(device)
        y = m(x)
        return y

    @unittest.skipIf(
        version.parse(torch_original.__version__) <= version.parse("1.13.0"),
        "deconv module don't support unbatched input in PyTorch before '1.13.0'",
    )
    @autotest(n=5)
    def test_deconv2d_auto_squeeze_with_random_data(test_case):
        channels = random(1, 6)
        m = torch.nn.ConvTranspose2d(
            in_channels=channels,
            out_channels=random(1, 20),
            kernel_size=random(1, 4),
            stride=random() | nothing(),
            padding=random(1, 3).to(int) | nothing(),
            dilation=random(1, 5) | nothing(),
            groups=random(1, 5) | nothing(),
            padding_mode=constant("zeros") | nothing(),
            bias=random_bool(),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=3, dim0=channels).to(device)
        y = m(x)
        return y

    @autotest(check_graph=False)
    def test_deconv2d_0size_with_random_data(test_case):
        channels = random(1, 6)
        m = torch.nn.ConvTranspose2d(
            in_channels=channels,
            out_channels=random(1, 20),
            kernel_size=random(1, 4),
            stride=random() | nothing(),
            padding=random(1, 3).to(int) | nothing(),
            dilation=random(1, 5) | nothing(),
            groups=random(1, 5) | nothing(),
            padding_mode=constant("zeros") | nothing(),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=4, dim0=0, dim1=channels).to(device)
        y = m(x)
        return y

    @unittest.skip(
        "Likely to fail the test. This case should run on cpu when the problem is solved."
    )
    @autotest(n=30, check_graph=False, rtol=1e-2, atol=1e-4)
    def test_deconv2d_group_with_random_data(test_case):
        channels = 720  # lcm(1, 2, 3, 4, 5, 6)
        m = torch.nn.ConvTranspose2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=random(1, 4),
            stride=random() | nothing(),
            padding=random(1, 3).to(int) | nothing(),
            dilation=random(1, 5) | nothing(),
            groups=random(1, 7),
            padding_mode=constant("zeros") | nothing(),
        )
        m.train(random())

        device = random_device()
        m.to(device)
        m.pytorch.to("cuda")
        x = random_tensor(ndim=4, dim1=channels).to(device)
        x.pytorch = x.pytorch.to("cuda")
        y = m(x)
        return y

    @profile(torch.nn.functional.conv_transpose2d)
    def profile_conv_transpose2d(test_case):
        inputs = torch.ones(16, 128, 128, 128)
        weights_4x4_64c = torch.ones(128, 64, 4, 4)
        weights_6x6_64c = torch.ones(128, 64, 6, 6)
        weights_8x8_64c = torch.ones(128, 64, 8, 8)
        torch.nn.functional.conv_transpose2d(
            inputs, weights_4x4_64c, stride=2, padding=1
        )
        torch.nn.functional.conv_transpose2d(
            inputs, weights_4x4_64c, stride=2, padding=1, bias=torch.ones(64)
        )
        torch.nn.functional.conv_transpose2d(
            inputs, weights_6x6_64c, stride=3, padding=2, output_padding=1
        )
        torch.nn.functional.conv_transpose2d(
            inputs,
            weights_6x6_64c,
            stride=3,
            padding=2,
            bias=torch.ones(64),
            output_padding=1,
        )
        torch.nn.functional.conv_transpose2d(
            inputs, weights_8x8_64c, stride=4, padding=2
        )
        torch.nn.functional.conv_transpose2d(
            inputs, weights_8x8_64c, stride=4, padding=2, bias=torch.ones(64)
        )


if __name__ == "__main__":
    unittest.main()
