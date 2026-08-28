| # | 通道名 | 组 | 量化上限 | 含义 |
|---|---|---|---|---|
| 0 | `hand_count1` | hand | 1.0 | 手牌中该牌种持有张数 >= 1（0/1） |
| 1 | `hand_count2` | hand | 1.0 | 手牌中该牌种持有张数 >= 2（0/1） |
| 2 | `hand_count3` | hand | 1.0 | 手牌中该牌种持有张数 >= 3（0/1） |
| 3 | `hand_count4` | hand | 1.0 | 手牌中该牌种持有张数 >= 4（0/1） |
| 4 | `hand_red5` | hand | 1.0 | 手牌含赤5（万/筒/索）标记 |
| 5 | `dora_indicators` | dora | 5.0 | 可见宝牌指示牌计数（含杠宝指示，最大5） |
| 6 | `dora_kinds` | dora | 4.0 | 指示牌对应的宝牌种类计数（上限4） |
| 7 | `last_draw` | basic | 1.0 | 刚摸的牌标记（摸后决策点） |
| 8 | `seat_wind` | basic | 1.0 | 自风 one-hot（列 27-30 = 东南西北） |
| 9 | `prevalent_wind` | basic | 1.0 | 场风 one-hot（东/南） |
| 10 | `round` | basic | 1.0 | 局数归一化（0..7 -> 0..1） |
| 11 | `honba` | basic | 1.0 | 本场数归一化 |
| 12 | `riichi_sticks` | basic | 1.0 | 立直棒数归一化 |
| 13 | `wall_left` | basic | 1.0 | 牌山剩余归一化（0..70） |
| 14 | `own_score_bucket` | basic | 1.0 | 自身点数 10k 分桶 one-hot |
| 15 | `rank` | basic | 1.0 | 自身顺位归一化（1位=0） |
| 16 | `is_oya` | basic | 1.0 | 是否亲家 |
| 17 | `visible_count` | visible | 1.0 | 全桌可见该牌种张数/4（手牌+副露+牌河+宝牌指示） |
| 18 | `river_s0_pos0` | river | 1.0 | 座位0 牌河第1张的牌种（one-hot；未切=0） |
| 19 | `river_handcut_s0_pos0` | river | 1.0 | 座位0 牌河第1张是否手切（1=手切/0=摸切） |
| 20 | `river_s0_pos1` | river | 1.0 | 座位0 牌河第2张的牌种（one-hot；未切=0） |
| 21 | `river_handcut_s0_pos1` | river | 1.0 | 座位0 牌河第2张是否手切（1=手切/0=摸切） |
| 22 | `river_s0_pos2` | river | 1.0 | 座位0 牌河第3张的牌种（one-hot；未切=0） |
| 23 | `river_handcut_s0_pos2` | river | 1.0 | 座位0 牌河第3张是否手切（1=手切/0=摸切） |
| 24 | `river_s0_pos3` | river | 1.0 | 座位0 牌河第4张的牌种（one-hot；未切=0） |
| 25 | `river_handcut_s0_pos3` | river | 1.0 | 座位0 牌河第4张是否手切（1=手切/0=摸切） |
| 26 | `river_s0_pos4` | river | 1.0 | 座位0 牌河第5张的牌种（one-hot；未切=0） |
| 27 | `river_handcut_s0_pos4` | river | 1.0 | 座位0 牌河第5张是否手切（1=手切/0=摸切） |
| 28 | `river_s0_pos5` | river | 1.0 | 座位0 牌河第6张的牌种（one-hot；未切=0） |
| 29 | `river_handcut_s0_pos5` | river | 1.0 | 座位0 牌河第6张是否手切（1=手切/0=摸切） |
| 30 | `river_s0_pos6` | river | 1.0 | 座位0 牌河第7张的牌种（one-hot；未切=0） |
| 31 | `river_handcut_s0_pos6` | river | 1.0 | 座位0 牌河第7张是否手切（1=手切/0=摸切） |
| 32 | `river_s0_pos7` | river | 1.0 | 座位0 牌河第8张的牌种（one-hot；未切=0） |
| 33 | `river_handcut_s0_pos7` | river | 1.0 | 座位0 牌河第8张是否手切（1=手切/0=摸切） |
| 34 | `river_s0_pos8` | river | 1.0 | 座位0 牌河第9张的牌种（one-hot；未切=0） |
| 35 | `river_handcut_s0_pos8` | river | 1.0 | 座位0 牌河第9张是否手切（1=手切/0=摸切） |
| 36 | `river_s0_pos9` | river | 1.0 | 座位0 牌河第10张的牌种（one-hot；未切=0） |
| 37 | `river_handcut_s0_pos9` | river | 1.0 | 座位0 牌河第10张是否手切（1=手切/0=摸切） |
| 38 | `river_s0_pos10` | river | 1.0 | 座位0 牌河第11张的牌种（one-hot；未切=0） |
| 39 | `river_handcut_s0_pos10` | river | 1.0 | 座位0 牌河第11张是否手切（1=手切/0=摸切） |
| 40 | `river_s0_pos11` | river | 1.0 | 座位0 牌河第12张的牌种（one-hot；未切=0） |
| 41 | `river_handcut_s0_pos11` | river | 1.0 | 座位0 牌河第12张是否手切（1=手切/0=摸切） |
| 42 | `river_s0_pos12` | river | 1.0 | 座位0 牌河第13张的牌种（one-hot；未切=0） |
| 43 | `river_handcut_s0_pos12` | river | 1.0 | 座位0 牌河第13张是否手切（1=手切/0=摸切） |
| 44 | `river_s0_pos13` | river | 1.0 | 座位0 牌河第14张的牌种（one-hot；未切=0） |
| 45 | `river_handcut_s0_pos13` | river | 1.0 | 座位0 牌河第14张是否手切（1=手切/0=摸切） |
| 46 | `river_s0_pos14` | river | 1.0 | 座位0 牌河第15张的牌种（one-hot；未切=0） |
| 47 | `river_handcut_s0_pos14` | river | 1.0 | 座位0 牌河第15张是否手切（1=手切/0=摸切） |
| 48 | `river_s0_pos15` | river | 1.0 | 座位0 牌河第16张的牌种（one-hot；未切=0） |
| 49 | `river_handcut_s0_pos15` | river | 1.0 | 座位0 牌河第16张是否手切（1=手切/0=摸切） |
| 50 | `river_s0_pos16` | river | 1.0 | 座位0 牌河第17张的牌种（one-hot；未切=0） |
| 51 | `river_handcut_s0_pos16` | river | 1.0 | 座位0 牌河第17张是否手切（1=手切/0=摸切） |
| 52 | `river_s0_pos17` | river | 1.0 | 座位0 牌河第18张的牌种（one-hot；未切=0） |
| 53 | `river_handcut_s0_pos17` | river | 1.0 | 座位0 牌河第18张是否手切（1=手切/0=摸切） |
| 54 | `river_s0_pos18` | river | 1.0 | 座位0 牌河第19张的牌种（one-hot；未切=0） |
| 55 | `river_handcut_s0_pos18` | river | 1.0 | 座位0 牌河第19张是否手切（1=手切/0=摸切） |
| 56 | `river_s0_pos19` | river | 1.0 | 座位0 牌河第20张的牌种（one-hot；未切=0） |
| 57 | `river_handcut_s0_pos19` | river | 1.0 | 座位0 牌河第20张是否手切（1=手切/0=摸切） |
| 58 | `river_s0_pos20` | river | 1.0 | 座位0 牌河第21张的牌种（one-hot；未切=0） |
| 59 | `river_handcut_s0_pos20` | river | 1.0 | 座位0 牌河第21张是否手切（1=手切/0=摸切） |
| 60 | `river_s0_pos21` | river | 1.0 | 座位0 牌河第22张的牌种（one-hot；未切=0） |
| 61 | `river_handcut_s0_pos21` | river | 1.0 | 座位0 牌河第22张是否手切（1=手切/0=摸切） |
| 62 | `river_s0_pos22` | river | 1.0 | 座位0 牌河第23张的牌种（one-hot；未切=0） |
| 63 | `river_handcut_s0_pos22` | river | 1.0 | 座位0 牌河第23张是否手切（1=手切/0=摸切） |
| 64 | `river_s0_pos23` | river | 1.0 | 座位0 牌河第24张的牌种（one-hot；未切=0） |
| 65 | `river_handcut_s0_pos23` | river | 1.0 | 座位0 牌河第24张是否手切（1=手切/0=摸切） |
| 66 | `river_s1_pos0` | river | 1.0 | 座位1 牌河第1张的牌种（one-hot；未切=0） |
| 67 | `river_handcut_s1_pos0` | river | 1.0 | 座位1 牌河第1张是否手切（1=手切/0=摸切） |
| 68 | `river_s1_pos1` | river | 1.0 | 座位1 牌河第2张的牌种（one-hot；未切=0） |
| 69 | `river_handcut_s1_pos1` | river | 1.0 | 座位1 牌河第2张是否手切（1=手切/0=摸切） |
| 70 | `river_s1_pos2` | river | 1.0 | 座位1 牌河第3张的牌种（one-hot；未切=0） |
| 71 | `river_handcut_s1_pos2` | river | 1.0 | 座位1 牌河第3张是否手切（1=手切/0=摸切） |
| 72 | `river_s1_pos3` | river | 1.0 | 座位1 牌河第4张的牌种（one-hot；未切=0） |
| 73 | `river_handcut_s1_pos3` | river | 1.0 | 座位1 牌河第4张是否手切（1=手切/0=摸切） |
| 74 | `river_s1_pos4` | river | 1.0 | 座位1 牌河第5张的牌种（one-hot；未切=0） |
| 75 | `river_handcut_s1_pos4` | river | 1.0 | 座位1 牌河第5张是否手切（1=手切/0=摸切） |
| 76 | `river_s1_pos5` | river | 1.0 | 座位1 牌河第6张的牌种（one-hot；未切=0） |
| 77 | `river_handcut_s1_pos5` | river | 1.0 | 座位1 牌河第6张是否手切（1=手切/0=摸切） |
| 78 | `river_s1_pos6` | river | 1.0 | 座位1 牌河第7张的牌种（one-hot；未切=0） |
| 79 | `river_handcut_s1_pos6` | river | 1.0 | 座位1 牌河第7张是否手切（1=手切/0=摸切） |
| 80 | `river_s1_pos7` | river | 1.0 | 座位1 牌河第8张的牌种（one-hot；未切=0） |
| 81 | `river_handcut_s1_pos7` | river | 1.0 | 座位1 牌河第8张是否手切（1=手切/0=摸切） |
| 82 | `river_s1_pos8` | river | 1.0 | 座位1 牌河第9张的牌种（one-hot；未切=0） |
| 83 | `river_handcut_s1_pos8` | river | 1.0 | 座位1 牌河第9张是否手切（1=手切/0=摸切） |
| 84 | `river_s1_pos9` | river | 1.0 | 座位1 牌河第10张的牌种（one-hot；未切=0） |
| 85 | `river_handcut_s1_pos9` | river | 1.0 | 座位1 牌河第10张是否手切（1=手切/0=摸切） |
| 86 | `river_s1_pos10` | river | 1.0 | 座位1 牌河第11张的牌种（one-hot；未切=0） |
| 87 | `river_handcut_s1_pos10` | river | 1.0 | 座位1 牌河第11张是否手切（1=手切/0=摸切） |
| 88 | `river_s1_pos11` | river | 1.0 | 座位1 牌河第12张的牌种（one-hot；未切=0） |
| 89 | `river_handcut_s1_pos11` | river | 1.0 | 座位1 牌河第12张是否手切（1=手切/0=摸切） |
| 90 | `river_s1_pos12` | river | 1.0 | 座位1 牌河第13张的牌种（one-hot；未切=0） |
| 91 | `river_handcut_s1_pos12` | river | 1.0 | 座位1 牌河第13张是否手切（1=手切/0=摸切） |
| 92 | `river_s1_pos13` | river | 1.0 | 座位1 牌河第14张的牌种（one-hot；未切=0） |
| 93 | `river_handcut_s1_pos13` | river | 1.0 | 座位1 牌河第14张是否手切（1=手切/0=摸切） |
| 94 | `river_s1_pos14` | river | 1.0 | 座位1 牌河第15张的牌种（one-hot；未切=0） |
| 95 | `river_handcut_s1_pos14` | river | 1.0 | 座位1 牌河第15张是否手切（1=手切/0=摸切） |
| 96 | `river_s1_pos15` | river | 1.0 | 座位1 牌河第16张的牌种（one-hot；未切=0） |
| 97 | `river_handcut_s1_pos15` | river | 1.0 | 座位1 牌河第16张是否手切（1=手切/0=摸切） |
| 98 | `river_s1_pos16` | river | 1.0 | 座位1 牌河第17张的牌种（one-hot；未切=0） |
| 99 | `river_handcut_s1_pos16` | river | 1.0 | 座位1 牌河第17张是否手切（1=手切/0=摸切） |
| 100 | `river_s1_pos17` | river | 1.0 | 座位1 牌河第18张的牌种（one-hot；未切=0） |
| 101 | `river_handcut_s1_pos17` | river | 1.0 | 座位1 牌河第18张是否手切（1=手切/0=摸切） |
| 102 | `river_s1_pos18` | river | 1.0 | 座位1 牌河第19张的牌种（one-hot；未切=0） |
| 103 | `river_handcut_s1_pos18` | river | 1.0 | 座位1 牌河第19张是否手切（1=手切/0=摸切） |
| 104 | `river_s1_pos19` | river | 1.0 | 座位1 牌河第20张的牌种（one-hot；未切=0） |
| 105 | `river_handcut_s1_pos19` | river | 1.0 | 座位1 牌河第20张是否手切（1=手切/0=摸切） |
| 106 | `river_s1_pos20` | river | 1.0 | 座位1 牌河第21张的牌种（one-hot；未切=0） |
| 107 | `river_handcut_s1_pos20` | river | 1.0 | 座位1 牌河第21张是否手切（1=手切/0=摸切） |
| 108 | `river_s1_pos21` | river | 1.0 | 座位1 牌河第22张的牌种（one-hot；未切=0） |
| 109 | `river_handcut_s1_pos21` | river | 1.0 | 座位1 牌河第22张是否手切（1=手切/0=摸切） |
| 110 | `river_s1_pos22` | river | 1.0 | 座位1 牌河第23张的牌种（one-hot；未切=0） |
| 111 | `river_handcut_s1_pos22` | river | 1.0 | 座位1 牌河第23张是否手切（1=手切/0=摸切） |
| 112 | `river_s1_pos23` | river | 1.0 | 座位1 牌河第24张的牌种（one-hot；未切=0） |
| 113 | `river_handcut_s1_pos23` | river | 1.0 | 座位1 牌河第24张是否手切（1=手切/0=摸切） |
| 114 | `river_s2_pos0` | river | 1.0 | 座位2 牌河第1张的牌种（one-hot；未切=0） |
| 115 | `river_handcut_s2_pos0` | river | 1.0 | 座位2 牌河第1张是否手切（1=手切/0=摸切） |
| 116 | `river_s2_pos1` | river | 1.0 | 座位2 牌河第2张的牌种（one-hot；未切=0） |
| 117 | `river_handcut_s2_pos1` | river | 1.0 | 座位2 牌河第2张是否手切（1=手切/0=摸切） |
| 118 | `river_s2_pos2` | river | 1.0 | 座位2 牌河第3张的牌种（one-hot；未切=0） |
| 119 | `river_handcut_s2_pos2` | river | 1.0 | 座位2 牌河第3张是否手切（1=手切/0=摸切） |
| 120 | `river_s2_pos3` | river | 1.0 | 座位2 牌河第4张的牌种（one-hot；未切=0） |
| 121 | `river_handcut_s2_pos3` | river | 1.0 | 座位2 牌河第4张是否手切（1=手切/0=摸切） |
| 122 | `river_s2_pos4` | river | 1.0 | 座位2 牌河第5张的牌种（one-hot；未切=0） |
| 123 | `river_handcut_s2_pos4` | river | 1.0 | 座位2 牌河第5张是否手切（1=手切/0=摸切） |
| 124 | `river_s2_pos5` | river | 1.0 | 座位2 牌河第6张的牌种（one-hot；未切=0） |
| 125 | `river_handcut_s2_pos5` | river | 1.0 | 座位2 牌河第6张是否手切（1=手切/0=摸切） |
| 126 | `river_s2_pos6` | river | 1.0 | 座位2 牌河第7张的牌种（one-hot；未切=0） |
| 127 | `river_handcut_s2_pos6` | river | 1.0 | 座位2 牌河第7张是否手切（1=手切/0=摸切） |
| 128 | `river_s2_pos7` | river | 1.0 | 座位2 牌河第8张的牌种（one-hot；未切=0） |
| 129 | `river_handcut_s2_pos7` | river | 1.0 | 座位2 牌河第8张是否手切（1=手切/0=摸切） |
| 130 | `river_s2_pos8` | river | 1.0 | 座位2 牌河第9张的牌种（one-hot；未切=0） |
| 131 | `river_handcut_s2_pos8` | river | 1.0 | 座位2 牌河第9张是否手切（1=手切/0=摸切） |
| 132 | `river_s2_pos9` | river | 1.0 | 座位2 牌河第10张的牌种（one-hot；未切=0） |
| 133 | `river_handcut_s2_pos9` | river | 1.0 | 座位2 牌河第10张是否手切（1=手切/0=摸切） |
| 134 | `river_s2_pos10` | river | 1.0 | 座位2 牌河第11张的牌种（one-hot；未切=0） |
| 135 | `river_handcut_s2_pos10` | river | 1.0 | 座位2 牌河第11张是否手切（1=手切/0=摸切） |
| 136 | `river_s2_pos11` | river | 1.0 | 座位2 牌河第12张的牌种（one-hot；未切=0） |
| 137 | `river_handcut_s2_pos11` | river | 1.0 | 座位2 牌河第12张是否手切（1=手切/0=摸切） |
| 138 | `river_s2_pos12` | river | 1.0 | 座位2 牌河第13张的牌种（one-hot；未切=0） |
| 139 | `river_handcut_s2_pos12` | river | 1.0 | 座位2 牌河第13张是否手切（1=手切/0=摸切） |
| 140 | `river_s2_pos13` | river | 1.0 | 座位2 牌河第14张的牌种（one-hot；未切=0） |
| 141 | `river_handcut_s2_pos13` | river | 1.0 | 座位2 牌河第14张是否手切（1=手切/0=摸切） |
| 142 | `river_s2_pos14` | river | 1.0 | 座位2 牌河第15张的牌种（one-hot；未切=0） |
| 143 | `river_handcut_s2_pos14` | river | 1.0 | 座位2 牌河第15张是否手切（1=手切/0=摸切） |
| 144 | `river_s2_pos15` | river | 1.0 | 座位2 牌河第16张的牌种（one-hot；未切=0） |
| 145 | `river_handcut_s2_pos15` | river | 1.0 | 座位2 牌河第16张是否手切（1=手切/0=摸切） |
| 146 | `river_s2_pos16` | river | 1.0 | 座位2 牌河第17张的牌种（one-hot；未切=0） |
| 147 | `river_handcut_s2_pos16` | river | 1.0 | 座位2 牌河第17张是否手切（1=手切/0=摸切） |
| 148 | `river_s2_pos17` | river | 1.0 | 座位2 牌河第18张的牌种（one-hot；未切=0） |
| 149 | `river_handcut_s2_pos17` | river | 1.0 | 座位2 牌河第18张是否手切（1=手切/0=摸切） |
| 150 | `river_s2_pos18` | river | 1.0 | 座位2 牌河第19张的牌种（one-hot；未切=0） |
| 151 | `river_handcut_s2_pos18` | river | 1.0 | 座位2 牌河第19张是否手切（1=手切/0=摸切） |
| 152 | `river_s2_pos19` | river | 1.0 | 座位2 牌河第20张的牌种（one-hot；未切=0） |
| 153 | `river_handcut_s2_pos19` | river | 1.0 | 座位2 牌河第20张是否手切（1=手切/0=摸切） |
| 154 | `river_s2_pos20` | river | 1.0 | 座位2 牌河第21张的牌种（one-hot；未切=0） |
| 155 | `river_handcut_s2_pos20` | river | 1.0 | 座位2 牌河第21张是否手切（1=手切/0=摸切） |
| 156 | `river_s2_pos21` | river | 1.0 | 座位2 牌河第22张的牌种（one-hot；未切=0） |
| 157 | `river_handcut_s2_pos21` | river | 1.0 | 座位2 牌河第22张是否手切（1=手切/0=摸切） |
| 158 | `river_s2_pos22` | river | 1.0 | 座位2 牌河第23张的牌种（one-hot；未切=0） |
| 159 | `river_handcut_s2_pos22` | river | 1.0 | 座位2 牌河第23张是否手切（1=手切/0=摸切） |
| 160 | `river_s2_pos23` | river | 1.0 | 座位2 牌河第24张的牌种（one-hot；未切=0） |
| 161 | `river_handcut_s2_pos23` | river | 1.0 | 座位2 牌河第24张是否手切（1=手切/0=摸切） |
| 162 | `river_s3_pos0` | river | 1.0 | 座位3 牌河第1张的牌种（one-hot；未切=0） |
| 163 | `river_handcut_s3_pos0` | river | 1.0 | 座位3 牌河第1张是否手切（1=手切/0=摸切） |
| 164 | `river_s3_pos1` | river | 1.0 | 座位3 牌河第2张的牌种（one-hot；未切=0） |
| 165 | `river_handcut_s3_pos1` | river | 1.0 | 座位3 牌河第2张是否手切（1=手切/0=摸切） |
| 166 | `river_s3_pos2` | river | 1.0 | 座位3 牌河第3张的牌种（one-hot；未切=0） |
| 167 | `river_handcut_s3_pos2` | river | 1.0 | 座位3 牌河第3张是否手切（1=手切/0=摸切） |
| 168 | `river_s3_pos3` | river | 1.0 | 座位3 牌河第4张的牌种（one-hot；未切=0） |
| 169 | `river_handcut_s3_pos3` | river | 1.0 | 座位3 牌河第4张是否手切（1=手切/0=摸切） |
| 170 | `river_s3_pos4` | river | 1.0 | 座位3 牌河第5张的牌种（one-hot；未切=0） |
| 171 | `river_handcut_s3_pos4` | river | 1.0 | 座位3 牌河第5张是否手切（1=手切/0=摸切） |
| 172 | `river_s3_pos5` | river | 1.0 | 座位3 牌河第6张的牌种（one-hot；未切=0） |
| 173 | `river_handcut_s3_pos5` | river | 1.0 | 座位3 牌河第6张是否手切（1=手切/0=摸切） |
| 174 | `river_s3_pos6` | river | 1.0 | 座位3 牌河第7张的牌种（one-hot；未切=0） |
| 175 | `river_handcut_s3_pos6` | river | 1.0 | 座位3 牌河第7张是否手切（1=手切/0=摸切） |
| 176 | `river_s3_pos7` | river | 1.0 | 座位3 牌河第8张的牌种（one-hot；未切=0） |
| 177 | `river_handcut_s3_pos7` | river | 1.0 | 座位3 牌河第8张是否手切（1=手切/0=摸切） |
| 178 | `river_s3_pos8` | river | 1.0 | 座位3 牌河第9张的牌种（one-hot；未切=0） |
| 179 | `river_handcut_s3_pos8` | river | 1.0 | 座位3 牌河第9张是否手切（1=手切/0=摸切） |
| 180 | `river_s3_pos9` | river | 1.0 | 座位3 牌河第10张的牌种（one-hot；未切=0） |
| 181 | `river_handcut_s3_pos9` | river | 1.0 | 座位3 牌河第10张是否手切（1=手切/0=摸切） |
| 182 | `river_s3_pos10` | river | 1.0 | 座位3 牌河第11张的牌种（one-hot；未切=0） |
| 183 | `river_handcut_s3_pos10` | river | 1.0 | 座位3 牌河第11张是否手切（1=手切/0=摸切） |
| 184 | `river_s3_pos11` | river | 1.0 | 座位3 牌河第12张的牌种（one-hot；未切=0） |
| 185 | `river_handcut_s3_pos11` | river | 1.0 | 座位3 牌河第12张是否手切（1=手切/0=摸切） |
| 186 | `river_s3_pos12` | river | 1.0 | 座位3 牌河第13张的牌种（one-hot；未切=0） |
| 187 | `river_handcut_s3_pos12` | river | 1.0 | 座位3 牌河第13张是否手切（1=手切/0=摸切） |
| 188 | `river_s3_pos13` | river | 1.0 | 座位3 牌河第14张的牌种（one-hot；未切=0） |
| 189 | `river_handcut_s3_pos13` | river | 1.0 | 座位3 牌河第14张是否手切（1=手切/0=摸切） |
| 190 | `river_s3_pos14` | river | 1.0 | 座位3 牌河第15张的牌种（one-hot；未切=0） |
| 191 | `river_handcut_s3_pos14` | river | 1.0 | 座位3 牌河第15张是否手切（1=手切/0=摸切） |
| 192 | `river_s3_pos15` | river | 1.0 | 座位3 牌河第16张的牌种（one-hot；未切=0） |
| 193 | `river_handcut_s3_pos15` | river | 1.0 | 座位3 牌河第16张是否手切（1=手切/0=摸切） |
| 194 | `river_s3_pos16` | river | 1.0 | 座位3 牌河第17张的牌种（one-hot；未切=0） |
| 195 | `river_handcut_s3_pos16` | river | 1.0 | 座位3 牌河第17张是否手切（1=手切/0=摸切） |
| 196 | `river_s3_pos17` | river | 1.0 | 座位3 牌河第18张的牌种（one-hot；未切=0） |
| 197 | `river_handcut_s3_pos17` | river | 1.0 | 座位3 牌河第18张是否手切（1=手切/0=摸切） |
| 198 | `river_s3_pos18` | river | 1.0 | 座位3 牌河第19张的牌种（one-hot；未切=0） |
| 199 | `river_handcut_s3_pos18` | river | 1.0 | 座位3 牌河第19张是否手切（1=手切/0=摸切） |
| 200 | `river_s3_pos19` | river | 1.0 | 座位3 牌河第20张的牌种（one-hot；未切=0） |
| 201 | `river_handcut_s3_pos19` | river | 1.0 | 座位3 牌河第20张是否手切（1=手切/0=摸切） |
| 202 | `river_s3_pos20` | river | 1.0 | 座位3 牌河第21张的牌种（one-hot；未切=0） |
| 203 | `river_handcut_s3_pos20` | river | 1.0 | 座位3 牌河第21张是否手切（1=手切/0=摸切） |
| 204 | `river_s3_pos21` | river | 1.0 | 座位3 牌河第22张的牌种（one-hot；未切=0） |
| 205 | `river_handcut_s3_pos21` | river | 1.0 | 座位3 牌河第22张是否手切（1=手切/0=摸切） |
| 206 | `river_s3_pos22` | river | 1.0 | 座位3 牌河第23张的牌种（one-hot；未切=0） |
| 207 | `river_handcut_s3_pos22` | river | 1.0 | 座位3 牌河第23张是否手切（1=手切/0=摸切） |
| 208 | `river_s3_pos23` | river | 1.0 | 座位3 牌河第24张的牌种（one-hot；未切=0） |
| 209 | `river_handcut_s3_pos23` | river | 1.0 | 座位3 牌河第24张是否手切（1=手切/0=摸切） |
| 210 | `discard_count_s0` | discards | 1.0 | 座位0 牌河该牌种张数/4 |
| 211 | `discard_riichi_s0` | discards | 1.0 | 座位0 牌河riichi标记 |
| 212 | `discard_red_s0` | discards | 1.0 | 座位0 牌河red标记 |
| 213 | `discard_count_s1` | discards | 1.0 | 座位1 牌河该牌种张数/4 |
| 214 | `discard_riichi_s1` | discards | 1.0 | 座位1 牌河riichi标记 |
| 215 | `discard_red_s1` | discards | 1.0 | 座位1 牌河red标记 |
| 216 | `discard_count_s2` | discards | 1.0 | 座位2 牌河该牌种张数/4 |
| 217 | `discard_riichi_s2` | discards | 1.0 | 座位2 牌河riichi标记 |
| 218 | `discard_red_s2` | discards | 1.0 | 座位2 牌河red标记 |
| 219 | `discard_count_s3` | discards | 1.0 | 座位3 牌河该牌种张数/4 |
| 220 | `discard_riichi_s3` | discards | 1.0 | 座位3 牌河riichi标记 |
| 221 | `discard_red_s3` | discards | 1.0 | 座位3 牌河red标记 |
| 222 | `meld_count_s0` | melds | 1.0 | 座位0 副露该牌种张数/4 |
| 223 | `meld_total_s0` | melds | 1.0 | 座位0 副露组数/4 |
| 224 | `meld_red_s0` | melds | 1.0 | 座位0 副露赤5标记 |
| 225 | `meld_type_s0` | melds | 1.0 | 座位0 副露类型（吃1/3、碰2/3、杠1） |
| 226 | `meld_count_s1` | melds | 1.0 | 座位1 副露该牌种张数/4 |
| 227 | `meld_total_s1` | melds | 1.0 | 座位1 副露组数/4 |
| 228 | `meld_red_s1` | melds | 1.0 | 座位1 副露赤5标记 |
| 229 | `meld_type_s1` | melds | 1.0 | 座位1 副露类型（吃1/3、碰2/3、杠1） |
| 230 | `meld_count_s2` | melds | 1.0 | 座位2 副露该牌种张数/4 |
| 231 | `meld_total_s2` | melds | 1.0 | 座位2 副露组数/4 |
| 232 | `meld_red_s2` | melds | 1.0 | 座位2 副露赤5标记 |
| 233 | `meld_type_s2` | melds | 1.0 | 座位2 副露类型（吃1/3、碰2/3、杠1） |
| 234 | `meld_count_s3` | melds | 1.0 | 座位3 副露该牌种张数/4 |
| 235 | `meld_total_s3` | melds | 1.0 | 座位3 副露组数/4 |
| 236 | `meld_red_s3` | melds | 1.0 | 座位3 副露赤5标记 |
| 237 | `meld_type_s3` | melds | 1.0 | 座位3 副露类型（吃1/3、碰2/3、杠1） |
| 238 | `n_kan` | state | 1.0 | 场上总杠数归一化 |
| 239 | `riichi_declared_s0` | state | 1.0 | 座位0 是否已立直 |
| 240 | `riichi_declared_s1` | state | 1.0 | 座位1 是否已立直 |
| 241 | `riichi_declared_s2` | state | 1.0 | 座位2 是否已立直 |
| 242 | `riichi_declared_s3` | state | 1.0 | 座位3 是否已立直 |
| 243 | `score_bucket_s0` | state | 1.0 | 座位0 点数 10k 分桶 one-hot |
| 244 | `score_bucket_s1` | state | 1.0 | 座位1 点数 10k 分桶 one-hot |
| 245 | `score_bucket_s2` | state | 1.0 | 座位2 点数 10k 分桶 one-hot |
| 246 | `score_bucket_s3` | state | 1.0 | 座位3 点数 10k 分桶 one-hot |
| 247 | `point_diff_0` | state | 1.0 | 与下家/对家/上家的点差归一化（+40000 平移至 0..1） |
| 248 | `point_diff_1` | state | 1.0 | 与下家/对家/上家的点差归一化（+40000 平移至 0..1） |
| 249 | `point_diff_2` | state | 1.0 | 与下家/对家/上家的点差归一化（+40000 平移至 0..1） |
| 250 | `last_event_type` | state | 1.0 | 上一事件类型列编码（摸/切/吃/碰/杠/立直/和） |
| 251 | `last_event_seat` | state | 1.0 | 上一事件来源座位/3 |
| 252 | `last_event_tile` | state | 1.0 | 上一事件涉及牌标记 |
| 253 | `riichi_sticks_onehot` | state | 1.0 | 立直棒数 one-hot（列=棒数） |
| 254 | `round_onehot` | state | 1.0 | 局数 one-hot（列 0..7 = E1..E4,S1..S4） |
| 255 | `honba_onehot` | state | 1.0 | 本场数 one-hot |
| 256 | `visible_red5` | state | 1.0 | 可见赤5总数/3（手牌+副露+牌河） |
| 257 | `discards_total_s0` | state | 1.0 | 座位0 已切牌数/24 |
| 258 | `discards_total_s1` | state | 1.0 | 座位1 已切牌数/24 |
| 259 | `discards_total_s2` | state | 1.0 | 座位2 已切牌数/24 |
| 260 | `discards_total_s3` | state | 1.0 | 座位3 已切牌数/24 |
| 261 | `shanten_current` | shanten | 1.0 | 当前向听数（-1..7 归一化到 0..1） |
| 262 | `shanten_discard` | shanten | 1.0 | 切X后的向听数（34列；不在手中的牌=当前向听数） |
| 263 | `shanten_ukeire` | shanten | 1.0 | 进张数/34（13张点；向听>2 时返回 0） |
| 264 | `shanten_waits` | shanten | 1.0 | 听牌时和了牌 0/1（34列；非听牌全 0） |
| 265 | `shanten_waits_count` | shanten | 1.0 | 听牌时和了牌种数/34（非听牌=0） |
| 266 | `shanten_waits_total` | shanten | 1.0 | 听牌时剩余和了张数/24（按全桌可见扣减；非听牌=0） |
| 267 | `uradora_expected` | state | 1.0 | 里宝期望：立直人数 × Σ(指示牌宝牌剩余张/牌山剩余)/4 |
| 268 | `shanten_regular` | shanten | 1.0 | 一般形向听数（-1..8 归一化 0..1） |
| 269 | `shanten_chiitoi` | shanten | 1.0 | 七对子向听数（-1..6 归一化 0..1） |
| 270 | `shanten_kokushi` | shanten | 1.0 | 国士无双向听数（-1..13 归一化 0..1） |
| 271 | `hand_dora` | hand | 1.0 | 手牌中是宝牌的牌种标记 |
| 272 | `dora_ind0_tile` | dora | 1.0 | 第1个宝牌指示牌是什么（one-hot；不足则全0） |
| 273 | `dora_ind0_dora` | dora | 1.0 | 第1个指示牌对应的宝牌是什么（one-hot） |
| 274 | `dora_ind1_tile` | dora | 1.0 | 第2个宝牌指示牌是什么（one-hot；不足则全0） |
| 275 | `dora_ind1_dora` | dora | 1.0 | 第2个指示牌对应的宝牌是什么（one-hot） |
| 276 | `dora_ind2_tile` | dora | 1.0 | 第3个宝牌指示牌是什么（one-hot；不足则全0） |
| 277 | `dora_ind2_dora` | dora | 1.0 | 第3个指示牌对应的宝牌是什么（one-hot） |
| 278 | `dora_ind3_tile` | dora | 1.0 | 第4个宝牌指示牌是什么（one-hot；不足则全0） |
| 279 | `dora_ind3_dora` | dora | 1.0 | 第4个指示牌对应的宝牌是什么（one-hot） |
| 280 | `dora_ind4_tile` | dora | 1.0 | 第5个宝牌指示牌是什么（one-hot；不足则全0） |
| 281 | `dora_ind4_dora` | dora | 1.0 | 第5个指示牌对应的宝牌是什么（one-hot） |
| 282 | `visible_dora_count` | visible | 1.0 | 全桌可见宝牌总张数/8（手牌+副露+牌河） |
| 283 | `hand_value_discard` | state | 1.0 | 切X后预期打点/32000（引擎计分；仅听牌；立直含里宝期望） |
