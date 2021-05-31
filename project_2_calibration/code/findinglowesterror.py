import os
import random
import shutil
import sys

from CameraCalibrator import CameraCalibrator
from VideoFrameExtractor import VideoFrameExtractor

chc1 = "videos/ch0.mp4"  # "videos/checkerboard_000.h264"
chc2 = "videos/ch1.mp4"  # "videos/checkerboard_019.h264"
chc_size = (7, 6)

# video 1
extractor_1 = VideoFrameExtractor(chc1)
extractor_2 = VideoFrameExtractor(chc2)
fids_1 = {
    (7, 7): [930, 931, 935, 936, 939, 950, 1177, 1178, 1179, 1378],
    (7, 6): [603, 604, 606, 607, 612, 613, 640, 760, 761, 762, 764, 770, 773, 817, 836, 837, 838, 845, 847, 848, 849, 904, 906, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 936, 938, 939, 941, 944, 954, 955, 956, 957, 958, 960, 963, 1018, 1019, 1020, 1021, 1176, 1178, 1180, 1186, 1187, 1188, 1258, 1343, 1344, 1345, 1346, 1347, 1363, 1365, 1366, 1367, 1368, 1369, 1370, 1376, 1377, 1378, 1379, 1380, 1384, 1386, 1415, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1623, 1625, 1684, 1685, 1686, 1754, 1755, 1757],
    (6, 7): [603, 604, 606, 607, 612, 613, 640, 760, 761, 762, 764, 770, 773, 817, 836, 837, 838, 845, 847, 848, 849, 904, 906, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 936, 938, 939, 941, 944, 954, 955, 956, 957, 958, 960, 963, 1018, 1019, 1020, 1021, 1176, 1178, 1180, 1186, 1187, 1188, 1258, 1343, 1344, 1345, 1346, 1347, 1363, 1365, 1366, 1367, 1368, 1369, 1370, 1376, 1377, 1378, 1379, 1380, 1384, 1386, 1415, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1623, 1625, 1684, 1685, 1686, 1754, 1755, 1757],
    (6, 6): [69, 70, 71, 72, 73, 92, 93, 138, 158, 160, 162, 178, 179, 182, 183, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 199, 200, 202, 203, 204, 206, 212, 213, 215, 216, 217, 218, 219, 220, 221, 222, 223, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 388, 398, 399, 400, 401, 402, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 469, 470, 471, 472, 473, 474, 475, 476, 477, 481, 487, 488, 489, 490, 491, 492, 493, 495, 496, 497, 500, 501, 502, 503, 517, 518, 519, 520, 521, 523, 540, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 623, 624, 625, 626, 627, 636, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 711, 712, 713, 714, 715, 722, 723, 724, 725, 726, 729, 730, 756, 757, 758, 763, 764, 773, 774, 776, 820, 821, 822, 835, 849, 850, 851, 856, 857, 858, 938, 939, 940, 941, 946, 947, 948, 949, 950, 951, 952, 953, 955, 956, 959, 960, 961, 962, 963, 972, 973, 974, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 992, 993, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1004, 1005, 1006, 1007, 1009, 1010, 1011, 1012, 1013, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1039, 1173, 1174, 1175, 1176, 1186, 1187, 1188, 1189, 1214, 1215, 1216, 1217, 1218, 1220, 1226, 1227, 1228, 1229, 1230, 1231, 1241, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1264, 1265, 1266, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1350, 1351, 1361, 1373, 1374, 1375, 1376, 1379, 1382, 1383, 1384, 1385, 1386, 1398, 1402, 1403, 1404, 1410, 1411, 1412, 1415, 1417, 1418, 1419, 1420, 1481, 1482, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1593, 1594, 1595, 1598, 1599, 1609, 1619, 1621, 1622, 1623, 1624, 1625, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1658, 1659, 1722, 1723, 1724, 1725, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1742, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783]
}

fids_2 = {
    (7, 7): [],
    (7, 6): [121, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 146, 147, 148, 149, 150, 176, 179, 670],
    (6, 7): [121, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 146, 147, 148, 149, 150, 176, 179, 670],
    (6, 6): [120, 121, 122, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 151, 152, 524, 525, 526, 621, 622, 623, 651, 652, 653, 654, 669, 674, 675, 676, 678, 679, 680, 681, 682, 683, 686, 688, 696, 697, 698, 704, 705, 706, 707, 708, 709, 710, 714, 715, 716, 717, 722, 723, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 739, 740, 741, 742, 743, 746]
}

# getting 10 random samples of frame ids
rfids_frames = []
sample_size = 10
for _ in range(0, 10):
    rfids_1 = random.sample(fids_1[chc_size], k=sample_size)
    frames_1 = extractor_1.get_video_frames(rfids_1)
    rfids_2 = random.sample(fids_2[chc_size], k=sample_size)
    frames_2 = extractor_2.get_video_frames(rfids_2)
    rfids_frames.append((rfids_1, rfids_2, frames_1 + frames_2))

# calculate the error for all of them
error_params_list = []
for f1, f2, fr in rfids_frames:
    calibrator = CameraCalibrator(fr, chc_size)
    params = calibrator.get_camera_params()
    print(f"Error: {params[0]}")
    error_params_list.append((f1, f2, fr, params[0], params))

# find the frames with lowest error
s_errs = sorted(error_params_list, key=lambda t: t[3])

print(f"Checkerboard: {chc_size}")
print(f"Lowest error: {s_errs[0][3]}")
print(f"V1: {s_errs[0][0]}")
print(f"V2: {s_errs[0][1]}")
params = s_errs[0][4]
m_frames = s_errs[0][2]

print(f"Error: {params[0]}")

# undistort images
subfolder = "calibration"
shutil.rmtree(subfolder, ignore_errors=True)
os.mkdir(subfolder)
for fid, frame in enumerate(m_frames):
    calibrator.undistort(frame, f"{subfolder}/img{fid}", params)

# Checkerboard: (7, 6)
# Lowest error: 0.42910958947918715
# V1: [838, 1755, 1617, 939, 1623, 1365, 913, 1685, 1621, 914]
# V2: [126, 148, 129, 147, 127, 130, 146, 150, 133, 176]
# Error: 0.42910958947918715

# Checkerboard: (6, 7)
# Lowest error: 0.43238291098606607
# V1: [922, 1347, 764, 1345, 954, 923, 1614, 1620, 1363, 912]
# V2: [132, 146, 148, 136, 121, 137, 670, 126, 134, 128]
# Error: 0.43238291098606607

# Checkerboard: (7, 6)
# Lowest error: 0.4469155752744077
# V1: [1616, 848, 1347, 938, 1343, 613, 925, 963, 1617, 1367]
# V2: [133, 136, 132, 129, 176, 148, 179, 147, 126, 135]
# Error: 0.4469155752744077

# all 6b6
# v1 [858, 189, 946, 1659, 1630, 1729, 1640, 1374, 940, 1411]
# v2 [621, 651, 697, 710, 709, 730, 696, 129, 741, 669]
# Error: 0.6565798273230387

#V1: [1231, 1256, 1645, 1258, 1545, 283, 1374, 980, 835, 1766]
#V2: [728, 122, 120, 731, 734, 705, 706, 708, 729, 680]
#Error: 0.4610511176676023

#V1: [1375, 1340, 1552, 276, 252, 1580, 278, 615, 262, 1595]
#V2: [129, 714, 681, 710, 128, 732, 741, 733, 728, 736]
#Error: 0.4841687136452362