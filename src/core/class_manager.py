import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from loguru import logger
from src.core.config import settings
from src.core.config_manager import config_manager
from src.api.schemas.classes import (
    CategoryItem,
    CategoryListResponse,
    SignClassItem,
    ClassifierClassesResponse,
)

DEFAULT_CLASSES: Dict[int, str] = {
    0: "No_entry",
    1: "No_cars",
    2: "No_cars_turn_right",
    3: "No_cars_turn_left",
    4: "No_motorcycles",
    5: "No_trucks",
    6: "No_trucks_over_2.5T",
    7: "No_pedestrians",
    8: "Height_limit",
    9: "No_turn_left",
    10: "No_turn_right",
    11: "No_U_turn",
    12: "No_turn_left_and_U_turn",
    13: "Speed_limit_40",
    14: "Speed_limit_50",
    15: "Speed_limit_60",
    16: "Speed_limit_70",
    17: "Speed_limit_80",
    18: "No_honking",
    19: "No_stopping_and_parking",
    20: "No_parking",
    21: "No_parking_odd_days",
    22: "No_parking_even_days",
    23: "No_straight_ahead",
    24: "No_turn_left_and_right",
    25: "Dangerous_curve_left",
    26: "Road_narrows",
    27: "Crossroad",
    28: "Intersection_with_non_priority_road",
    29: "Give_way",
    30: "Road_bump",
    31: "Pedestrian_crossing",
    32: "Children_crossing",
    33: "Road_works",
    34: "Overhead_power_lines",
    35: "Clearance_height",
    36: "Slow_down",
    37: "Go_straight",
    38: "Keep_right",
    39: "Roundabout",
    40: "Main_road",
    41: "End_main_road",
    42: "Pedestrian_crossing_zone",
    43: "Hospital",
    44: "Bus_stop",
    45: "Market",
    46: "Other_prohibition_signs",
    47: "No_slow_moving_vehicles",
    48: "U_turn_area",
    49: "Pedestrian_overpass",
    50: "Width_limit",
    51: "Weight_limit",
    52: "No_trailers",
    53: "No_vehicles_carrying_dangerous_goods",
    54: "Tunnel",
    55: "No_bicycles",
    56: "Cliff_ahead",
    57: "Min_speed_limit_30",
    58: "Min_distance_between_vehicles",
    59: "No_overtaking",
    60: "End_min_speed_limit_30",
    61: "No_motorcycles_and_slow_vehicles",
    62: "End_dual_carriageway",
    63: "No_buses_and_trucks",
    64: "Two_way_traffic",
    65: "Accident_area",
    66: "Turn_left_only",
    67: "Turn_right_only",
    68: "One_way_road",
    69: "Railway_crossing_with_gates",
    70: "Underpass",
    71: "Residential_area",
    72: "End_residential_area",
    73: "No_cars_U_turn",
    74: "Dangerous_curve_right",
    75: "Multiple_curves",
    76: "Road_narrows_left",
    77: "Road_narrows_right",
    78: "Intersection_with_non_priority_road_right",
    79: "T_junction",
    80: "T_junction_right",
    81: "T_junction_left",
    82: "Intersection_with_non_priority_road_left",
    83: "End_no_overtaking",
    84: "End_all_prohibitions",
    85: "End_tunnel",
    86: "Speed_limit_90",
    87: "Speed_limit_100",
    88: "Road_reserved_for_cars",
    89: "Expressway",
    90: "Axle_weight_limit",
    91: "Road_closed",
    92: "Dead_end",
    93: "Reserved_for_disabled",
    94: "Checkpoint",
    95: "Speed_limit_120",
    96: "Parking_area",
    97: "Min_speed_limit_60",
    98: "Steep_descent_5_percent",
    99: "Steep_ascent_5_percent",
}

DEFAULT_CATEGORIES: List[Dict[str, Any]] = [
        {
            "id": 1,
            "code": "MOCK_TRAFFIC_SIGNS",
            "name_vi": "Mock Traffic Signs",
            "name_en": "Mock Traffic Signs",
            "description": "Preview test data"
        },
        {
            "id": 2,
            "code": "WARNING",
            "name_vi": "Biển báo nguy hiểm và cảnh báo",
            "name_en": "Warning sign",
            "description": "Nhóm biển báo nguy hiểm (W) theo QCVN 41:2019"
        },
        {
            "id": 3,
            "code": "MANDATORY",
            "name_vi": "Biển hiệu lệnh",
            "name_en": "Mandatory sign",
            "description": "Nhóm biển hiệu lệnh (R) theo QCVN 41:2019"
        },
        {
            "id": 4,
            "code": "SPEED_LIMIT",
            "name_vi": "Biển giới hạn tốc độ",
            "name_en": "Speed limit sign",
            "description": "Nhóm biển giới hạn tốc độ theo QCVN 41:2019"
        },
        {
            "id": 5,
            "code": "TEMPORARY",
            "name_vi": "Biển báo tạm thời/công trường",
            "name_en": "Temporary/roadwork sign",
            "description": "Biển công trường theo QCVN 41:2019"
        },
        {
            "id": 6,
            "code": "PROHIBITORY",
            "name_vi": "Biển báo cấm",
            "name_en": "Prohibitory sign",
            "description": "Nhóm biển báo cấm (P) theo QCVN 41:2019"
        },
        {
            "id": 7,
            "code": "INFORMATION",
            "name_vi": "Biển chỉ dẫn",
            "name_en": "Information sign",
            "description": "Nhóm biển chỉ dẫn (I) theo QCVN 41:2019"
        }
    ]

DEFAULT_SIGN_METADATA: Dict[int, Dict[str, Any]] = {
    0: {"category": 'PROHIBITORY', "name_vi": 'Đường cấm / Cấm đi ngược chiều', "name_en": 'No entry', "standard_code": 'P.102'},
    1: {"category": 'PROHIBITORY', "name_vi": 'Cấm xe ô tô', "name_en": 'No cars', "standard_code": 'P.103a'},
    2: {"category": 'PROHIBITORY', "name_vi": 'Cấm xe ô tô rẽ phải', "name_en": 'No cars turn right', "standard_code": 'P.103b'},
    3: {"category": 'PROHIBITORY', "name_vi": 'Cấm xe ô tô rẽ trái', "name_en": 'No cars turn left', "standard_code": 'P.103c'},
    4: {"category": 'PROHIBITORY', "name_vi": 'Cấm xe mô tô', "name_en": 'No motorcycles', "standard_code": 'P.104'},
    5: {"category": 'PROHIBITORY', "name_vi": 'Cấm xe ô tô tải', "name_en": 'No trucks', "standard_code": 'P.106a'},
    6: {"category": 'PROHIBITORY', "name_vi": 'Cấm xe ô tô tải trên 2.5 tấn', "name_en": 'No trucks over 2.5T', "standard_code": 'P.106b'},
    7: {"category": 'PROHIBITORY', "name_vi": 'Cấm người đi bộ', "name_en": 'No pedestrians', "standard_code": 'P.112'},
    8: {"category": 'PROHIBITORY', "name_vi": 'Hạn chế chiều cao', "name_en": 'Height limit', "standard_code": 'P.117'},
    9: {"category": 'PROHIBITORY', "name_vi": 'Cấm rẽ trái', "name_en": 'No turn left', "standard_code": 'P.123a'},
    10: {"category": 'PROHIBITORY', "name_vi": 'Cấm rẽ phải', "name_en": 'No turn right', "standard_code": 'P.123b'},
    11: {"category": 'PROHIBITORY', "name_vi": 'Cấm quay đầu xe', "name_en": 'No U turn', "standard_code": 'P.124a'},
    12: {"category": 'PROHIBITORY', "name_vi": 'Cấm rẽ trái và quay đầu xe', "name_en": 'No turn left and U turn', "standard_code": 'P.124c'},
    13: {"category": 'SPEED_LIMIT', "name_vi": 'Tốc độ tối đa cho phép 40 km/h', "name_en": 'Speed limit 40', "standard_code": 'P.127'},
    14: {"category": 'SPEED_LIMIT', "name_vi": 'Tốc độ tối đa cho phép 50 km/h', "name_en": 'Speed limit 50', "standard_code": 'P.127'},
    15: {"category": 'SPEED_LIMIT', "name_vi": 'Tốc độ tối đa cho phép 60 km/h', "name_en": 'Speed limit 60', "standard_code": 'P.127'},
    16: {"category": 'SPEED_LIMIT', "name_vi": 'Tốc độ tối đa cho phép 70 km/h', "name_en": 'Speed limit 70', "standard_code": 'P.127'},
    17: {"category": 'SPEED_LIMIT', "name_vi": 'Tốc độ tối đa cho phép 80 km/h', "name_en": 'Speed limit 80', "standard_code": 'P.127'},
    18: {"category": 'PROHIBITORY', "name_vi": 'Cấm bấm còi', "name_en": 'No honking', "standard_code": 'P.128'},
    19: {"category": 'PROHIBITORY', "name_vi": 'Cấm dừng xe và đỗ xe', "name_en": 'No stopping and parking', "standard_code": 'P.130'},
    20: {"category": 'PROHIBITORY', "name_vi": 'Cấm đỗ xe', "name_en": 'No parking', "standard_code": 'P.131a'},
    21: {"category": 'PROHIBITORY', "name_vi": 'Cấm đỗ xe ngày lẻ', "name_en": 'No parking odd days', "standard_code": 'P.131b'},
    22: {"category": 'PROHIBITORY', "name_vi": 'Cấm đỗ xe ngày chẵn', "name_en": 'No parking even days', "standard_code": 'P.131c'},
    23: {"category": 'PROHIBITORY', "name_vi": 'Cấm đi thẳng', "name_en": 'No straight ahead', "standard_code": 'P.137'},
    24: {"category": 'PROHIBITORY', "name_vi": 'Cấm rẽ trái và rẽ phải', "name_en": 'No turn left and right', "standard_code": 'P.138'},
    25: {"category": 'WARNING', "name_vi": 'Chỗ ngoặt nguy hiểm vòng bên trái', "name_en": 'Dangerous curve left', "standard_code": 'W.201a'},
    26: {"category": 'WARNING', "name_vi": 'Đường bị hẹp cả hai bên', "name_en": 'Road narrows', "standard_code": 'W.203a'},
    27: {"category": 'WARNING', "name_vi": 'Giao nhau cùng mức các tuyến đường cùng cấp', "name_en": 'Crossroad', "standard_code": 'W.205a'},
    28: {"category": 'WARNING', "name_vi": 'Giao nhau với đường không ưu tiên', "name_en": 'Intersection with non priority road', "standard_code": 'W.207a'},
    29: {"category": 'WARNING', "name_vi": 'Giao nhau với đường ưu tiên', "name_en": 'Give way', "standard_code": 'W.208'},
    30: {"category": 'WARNING', "name_vi": 'Đường có gồ giảm tốc / mấp mô', "name_en": 'Road bump', "standard_code": 'W.221b'},
    31: {"category": 'WARNING', "name_vi": 'Đoạn đường người đi bộ cắt ngang', "name_en": 'Pedestrian crossing', "standard_code": 'W.224'},
    32: {"category": 'WARNING', "name_vi": 'Đoạn đường hay có trẻ em ngang qua', "name_en": 'Children crossing', "standard_code": 'W.225'},
    33: {"category": 'WARNING', "name_vi": 'Công trường đang thi công', "name_en": 'Road works', "standard_code": 'W.227'},
    34: {"category": 'WARNING', "name_vi": 'Đường cáp điện ở phía trên', "name_en": 'Overhead power lines', "standard_code": 'W.239a'},
    35: {"category": 'WARNING', "name_vi": 'Chiều cao tĩnh không thực tế', "name_en": 'Clearance height', "standard_code": 'W.239b'},
    36: {"category": 'WARNING', "name_vi": 'Đi chậm', "name_en": 'Slow down', "standard_code": 'W.245a'},
    37: {"category": 'MANDATORY', "name_vi": 'Hướng đi thẳng phải theo', "name_en": 'Go straight', "standard_code": 'R.301a'},
    38: {"category": 'MANDATORY', "name_vi": 'Hướng đi phải theo: Đi về phía bên phải', "name_en": 'Keep right', "standard_code": 'R.301b'},
    39: {"category": 'MANDATORY', "name_vi": 'Nơi giao nhau chạy theo vòng xuyến', "name_en": 'Roundabout', "standard_code": 'R.303'},
    40: {"category": 'INFORMATION', "name_vi": 'Bắt đầu đường ưu tiên', "name_en": 'Main road', "standard_code": 'I.401'},
    41: {"category": 'INFORMATION', "name_vi": 'Hết đoạn đường ưu tiên', "name_en": 'End main road', "standard_code": 'I.402'},
    42: {"category": 'INFORMATION', "name_vi": 'Vị trí người đi bộ sang đường', "name_en": 'Pedestrian crossing zone', "standard_code": 'I.423a'},
    43: {"category": 'INFORMATION', "name_vi": 'Bệnh viện', "name_en": 'Hospital', "standard_code": 'I.428'},
    44: {"category": 'INFORMATION', "name_vi": 'Trạm dừng xe buýt', "name_en": 'Bus stop', "standard_code": 'I.434a'},
    45: {"category": 'WARNING', "name_vi": 'Đoạn đường hay họp chợ', "name_en": 'Market', "standard_code": 'W.234'},
    46: {"category": 'PROHIBITORY', "name_vi": 'Biển báo cấm khác / Cấm chung', "name_en": 'Other prohibition signs', "standard_code": 'P.140'},
    47: {"category": 'PROHIBITORY', "name_vi": 'Cấm xe súc vật kéo / xe thô sơ', "name_en": 'No slow moving vehicles', "standard_code": 'P.113'},
    48: {"category": 'INFORMATION', "name_vi": 'Khu vực quay đầu xe', "name_en": 'U turn area', "standard_code": 'I.409'},
    49: {"category": 'INFORMATION', "name_vi": 'Cầu vượt qua đường cho người đi bộ', "name_en": 'Pedestrian overpass', "standard_code": 'I.424a'},
    50: {"category": 'PROHIBITORY', "name_vi": 'Hạn chế chiều ngang xe', "name_en": 'Width limit', "standard_code": 'P.118'},
    51: {"category": 'PROHIBITORY', "name_vi": 'Hạn chế trọng lượng xe', "name_en": 'Weight limit', "standard_code": 'P.115'},
    52: {"category": 'PROHIBITORY', "name_vi": 'Cấm xe kéo rơ-moóc', "name_en": 'No trailers', "standard_code": 'P.108'},
    53: {"category": 'PROHIBITORY', "name_vi": 'Cấm xe chở hàng nguy hiểm', "name_en": 'No vehicles carrying dangerous goods', "standard_code": 'P.109'},
    54: {"category": 'WARNING', "name_vi": 'Đường hầm', "name_en": 'Tunnel', "standard_code": 'W.240'},
    55: {"category": 'PROHIBITORY', "name_vi": 'Cấm người đi xe đạp', "name_en": 'No bicycles', "standard_code": 'P.110a'},
    56: {"category": 'WARNING', "name_vi": 'Vách núi / Kè vực sâu nguy hiểm', "name_en": 'Cliff ahead', "standard_code": 'W.216a'},
    57: {"category": 'MANDATORY', "name_vi": 'Tốc độ tối thiểu cho phép 30 km/h', "name_en": 'Min speed limit 30', "standard_code": 'R.306'},
    58: {"category": 'PROHIBITORY', "name_vi": 'Cự ly tối thiểu giữa hai xe', "name_en": 'Min distance between vehicles', "standard_code": 'P.121'},
    59: {"category": 'PROHIBITORY', "name_vi": 'Cấm vượt', "name_en": 'No overtaking', "standard_code": 'P.125'},
    60: {"category": 'MANDATORY', "name_vi": 'Hết tốc độ tối thiểu cho phép 30 km/h', "name_en": 'End min speed limit 30', "standard_code": 'R.307'},
    61: {"category": 'PROHIBITORY', "name_vi": 'Cấm xe mô tô và xe thô sơ', "name_en": 'No motorcycles and slow vehicles', "standard_code": 'P.105'},
    62: {"category": 'WARNING', "name_vi": 'Hết đường đôi', "name_en": 'End dual carriageway', "standard_code": 'W.235'},
    63: {"category": 'PROHIBITORY', "name_vi": 'Cấm xe buýt và xe ô tô tải', "name_en": 'No buses and trucks', "standard_code": 'P.107'},
    64: {"category": 'WARNING', "name_vi": 'Đường hai chiều', "name_en": 'Two way traffic', "standard_code": 'W.204'},
    65: {"category": 'WARNING', "name_vi": 'Đoạn đường hay xảy ra tai nạn', "name_en": 'Accident area', "standard_code": 'W.244'},
    66: {"category": 'MANDATORY', "name_vi": 'Hướng đi phải theo: Rẽ trái', "name_en": 'Turn left only', "standard_code": 'R.301e'},
    67: {"category": 'MANDATORY', "name_vi": 'Hướng đi phải theo: Rẽ phải', "name_en": 'Turn right only', "standard_code": 'R.301d'},
    68: {"category": 'INFORMATION', "name_vi": 'Đường một chiều', "name_en": 'One way road', "standard_code": 'I.407a'},
    69: {"category": 'WARNING', "name_vi": 'Giao nhau với đường sắt có rào chắn', "name_en": 'Railway crossing with gates', "standard_code": 'W.210'},
    70: {"category": 'INFORMATION', "name_vi": 'Hầm chui cho người đi bộ', "name_en": 'Underpass', "standard_code": 'I.424b'},
    71: {"category": 'INFORMATION', "name_vi": 'Bắt đầu khu đông dân cư', "name_en": 'Residential area', "standard_code": 'R.420'},
    72: {"category": 'INFORMATION', "name_vi": 'Hết khu đông dân cư', "name_en": 'End residential area', "standard_code": 'R.421'},
    73: {"category": 'PROHIBITORY', "name_vi": 'Cấm xe ô tô quay đầu xe', "name_en": 'No cars U turn', "standard_code": 'P.124b'},
    74: {"category": 'WARNING', "name_vi": 'Chỗ ngoặt nguy hiểm vòng bên phải', "name_en": 'Dangerous curve right', "standard_code": 'W.201b'},
    75: {"category": 'WARNING', "name_vi": 'Chỗ ngoặt nguy hiểm liên tiếp', "name_en": 'Multiple curves', "standard_code": 'W.202'},
    76: {"category": 'WARNING', "name_vi": 'Đường bị hẹp về phía trái', "name_en": 'Road narrows left', "standard_code": 'W.203b'},
    77: {"category": 'WARNING', "name_vi": 'Đường bị hẹp về phía phải', "name_en": 'Road narrows right', "standard_code": 'W.203c'},
    78: {"category": 'WARNING', "name_vi": 'Giao nhau với đường không ưu tiên bên phải', "name_en": 'Intersection with non priority road right', "standard_code": 'W.207b'},
    79: {"category": 'WARNING', "name_vi": 'Giao nhau ngã ba chữ T', "name_en": 'T junction', "standard_code": 'W.205b'},
    80: {"category": 'WARNING', "name_vi": 'Giao nhau nhánh rẽ phải', "name_en": 'T junction right', "standard_code": 'W.205c'},
    81: {"category": 'WARNING', "name_vi": 'Giao nhau nhánh rẽ trái', "name_en": 'T junction left', "standard_code": 'W.205d'},
    82: {"category": 'WARNING', "name_vi": 'Giao nhau với đường không ưu tiên bên trái', "name_en": 'Intersection with non priority road left', "standard_code": 'W.207c'},
    83: {"category": 'PROHIBITORY', "name_vi": 'Hết cấm vượt', "name_en": 'End no overtaking', "standard_code": 'DP.133'},
    84: {"category": 'PROHIBITORY', "name_vi": 'Hết tất cả các lệnh cấm', "name_en": 'End all prohibitions', "standard_code": 'DP.135'},
    85: {"category": 'INFORMATION', "name_vi": 'Hết đường hầm', "name_en": 'End tunnel', "standard_code": 'I.441a'},
    86: {"category": 'SPEED_LIMIT', "name_vi": 'Tốc độ tối đa cho phép 90 km/h', "name_en": 'Speed limit 90', "standard_code": 'P.127'},
    87: {"category": 'SPEED_LIMIT', "name_vi": 'Tốc độ tối đa cho phép 100 km/h', "name_en": 'Speed limit 100', "standard_code": 'P.127'},
    88: {"category": 'MANDATORY', "name_vi": 'Đường dành cho xe ô tô', "name_en": 'Road reserved for cars', "standard_code": 'R.403a'},
    89: {"category": 'INFORMATION', "name_vi": 'Đường cao tốc', "name_en": 'Expressway', "standard_code": 'I.437'},
    90: {"category": 'PROHIBITORY', "name_vi": 'Hạn chế tải trọng trên trục xe', "name_en": 'Axle weight limit', "standard_code": 'P.116'},
    91: {"category": 'PROHIBITORY', "name_vi": 'Đường cấm tất cả các loại phương tiện', "name_en": 'Road closed', "standard_code": 'P.101'},
    92: {"category": 'INFORMATION', "name_vi": 'Đường cụt', "name_en": 'Dead end', "standard_code": 'I.405a'},
    93: {"category": 'INFORMATION', "name_vi": 'Vị trí dành riêng cho người khuyết tật', "name_en": 'Reserved for disabled', "standard_code": 'I.444'},
    94: {"category": 'PROHIBITORY', "name_vi": 'Trạm kiểm soát', "name_en": 'Checkpoint', "standard_code": 'P.129'},
    95: {"category": 'SPEED_LIMIT', "name_vi": 'Tốc độ tối đa cho phép 120 km/h', "name_en": 'Speed limit 120', "standard_code": 'P.127'},
    96: {"category": 'INFORMATION', "name_vi": 'Nơi đỗ xe', "name_en": 'Parking area', "standard_code": 'I.408'},
    97: {"category": 'MANDATORY', "name_vi": 'Tốc độ tối thiểu cho phép 60 km/h', "name_en": 'Min speed limit 60', "standard_code": 'R.306'},
    98: {"category": 'WARNING', "name_vi": 'Dốc xuống nguy hiểm 5%', "name_en": 'Steep descent 5 percent', "standard_code": 'W.219'},
    99: {"category": 'WARNING', "name_vi": 'Dốc lên nguy hiểm 5%', "name_en": 'Steep ascent 5 percent', "standard_code": 'W.220'},
}


class ClassManager:
    def __init__(self):
        self._base_dir = Path(__file__).resolve().parent.parent.parent
        self._classes_file = self._base_dir / "configs" / "classes.json"
        self._metadata_file = self._base_dir / "configs" / "sign_metadata.json"
        self._categories_file = self._base_dir / "configs" / "categories.json"
        self._classes: Dict[int, str] = {}
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._categories: List[Dict[str, Any]] = []
        self.load_classes()

    def load_classes(self) -> None:
        if self._classes_file.exists():
            try:
                with open(self._classes_file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    self._classes = {int(k): str(v) for k, v in raw_data.items()}
                logger.info(f"Loaded {len(self._classes)} classes from {self._classes_file}")
            except Exception as e:
                logger.warning(f"Failed to read classes from {self._classes_file}: {e}. Falling back to defaults.")
                self._classes = dict(DEFAULT_CLASSES)
        else:
            self._classes = dict(DEFAULT_CLASSES)

        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "r", encoding="utf-8") as f:
                    raw_meta = json.load(f)
                    self._metadata = {
                        int(k): {
                            "category": v.get("category"),
                            "name_vi": v.get("name_vi"),
                            "name_en": v.get("name_en"),
                            "standard_code": v.get("standard_code"),
                        }
                        for k, v in raw_meta.items()
                    }
                logger.info(f"Loaded {len(self._metadata)} sign metadata entries from {self._metadata_file}")
            except Exception as e:
                logger.warning(f"Failed to read metadata from {self._metadata_file}: {e}. Falling back to defaults.")
                self._metadata = {k: dict(v) for k, v in DEFAULT_SIGN_METADATA.items()}
        else:
            self._metadata = {k: dict(v) for k, v in DEFAULT_SIGN_METADATA.items()}

        if self._categories_file.exists():
            try:
                with open(self._categories_file, "r", encoding="utf-8") as f:
                    self._categories = json.load(f)
                logger.info(f"Loaded {len(self._categories)} categories from {self._categories_file}")
            except Exception as e:
                logger.warning(f"Failed to read categories from {self._categories_file}: {e}. Falling back to defaults.")
                self._categories = [dict(item) for item in DEFAULT_CATEGORIES]
        else:
            self._categories = [dict(item) for item in DEFAULT_CATEGORIES]

        self.save_classes()

    def save_classes(self) -> None:
        try:
            self._classes_file.parent.mkdir(parents=True, exist_ok=True)
            serializable_classes = {str(k): v for k, v in sorted(self._classes.items())}
            with open(self._classes_file, "w", encoding="utf-8") as f:
                json.dump(serializable_classes, f, indent=2, ensure_ascii=False)

            serializable_meta = {
                str(k): {
                    "class_id": k,
                    "class_name": self._classes.get(k, ""),
                    "category": v.get("category"),
                    "name_vi": v.get("name_vi"),
                    "name_en": v.get("name_en"),
                    "standard_code": v.get("standard_code"),
                }
                for k, v in sorted(self._metadata.items())
            }
            with open(self._metadata_file, "w", encoding="utf-8") as f:
                json.dump(serializable_meta, f, indent=2, ensure_ascii=False)

            with open(self._categories_file, "w", encoding="utf-8") as f:
                json.dump(self._categories, f, indent=2, ensure_ascii=False)

            logger.debug(f"Persisted classes, metadata, and categories configs")
        except Exception as e:
            logger.error(f"Failed to persist classes configurations: {e}")

    def get_classes_map(self) -> Dict[int, str]:
        return dict(self._classes)

    def get_sign_metadata(self, class_identifier: Union[int, str]) -> Dict[str, Any]:
        if isinstance(class_identifier, int):
            return dict(self._metadata.get(class_identifier, {}))
        target_name = str(class_identifier).strip()
        for cid, name in self._classes.items():
            if name == target_name:
                return dict(self._metadata.get(cid, {}))
        return {}

    def get_all_categories_response(self) -> CategoryListResponse:
        items = [CategoryItem(**item) for item in self._categories]
        return CategoryListResponse(
            total_categories=len(items),
            categories=items,
        )

    def format_item(self, class_id: int, class_name: str, template: Optional[str] = None) -> SignClassItem:
        active_template = template or config_manager.classifier_prompt_template or settings.CLASSIFIER_PROMPT_TEMPLATE
        human_readable = class_name.replace("_", " ")
        prompt = active_template.format(human_readable)
        meta = self._metadata.get(class_id, {})
        return SignClassItem(
            class_id=class_id,
            class_name=class_name,
            human_readable_name=human_readable,
            prompt=prompt,
            category=meta.get("category"),
            name_vi=meta.get("name_vi"),
            name_en=meta.get("name_en"),
            standard_code=meta.get("standard_code"),
        )

    def get_all_classes_response(self, template: Optional[str] = None) -> ClassifierClassesResponse:
        active_template = template or config_manager.classifier_prompt_template or settings.CLASSIFIER_PROMPT_TEMPLATE
        items = [
            self.format_item(cid, self._classes[cid], active_template)
            for cid in sorted(self._classes.keys())
        ]
        return ClassifierClassesResponse(
            total_classes=len(items),
            prompt_template=active_template,
            classes=items,
        )

    def get_class_by_id(self, class_id: int, template: Optional[str] = None) -> Optional[SignClassItem]:
        if class_id not in self._classes:
            return None
        return self.format_item(class_id, self._classes[class_id], template)

    def add_class(
        self,
        class_name: str,
        custom_id: Optional[int] = None,
        category: Optional[str] = None,
        name_vi: Optional[str] = None,
        name_en: Optional[str] = None,
        standard_code: Optional[str] = None,
        template: Optional[str] = None,
    ) -> SignClassItem:
        clean_name = class_name.strip().replace(" ", "_")
        if custom_id is not None:
            if custom_id in self._classes:
                raise ValueError(f"Class ID {custom_id} already exists with name '{self._classes[custom_id]}'.")
            target_id = custom_id
        else:
            target_id = max(self._classes.keys()) + 1 if self._classes else 0

        self._classes[target_id] = clean_name
        self._metadata[target_id] = {
            "category": category,
            "name_vi": name_vi,
            "name_en": name_en,
            "standard_code": standard_code,
        }
        self.save_classes()
        logger.info(f"Added class ID {target_id}: '{clean_name}' (category={category}, standard={standard_code})")
        return self.format_item(target_id, clean_name, template)

    def update_class(
        self,
        class_id: int,
        class_name: Optional[str] = None,
        category: Optional[str] = None,
        name_vi: Optional[str] = None,
        name_en: Optional[str] = None,
        standard_code: Optional[str] = None,
        template: Optional[str] = None,
    ) -> SignClassItem:
        if class_id not in self._classes:
            raise KeyError(f"Class ID {class_id} not found.")
        if class_name is not None:
            clean_name = class_name.strip().replace(" ", "_")
            self._classes[class_id] = clean_name
        else:
            clean_name = self._classes[class_id]

        if class_id not in self._metadata:
            self._metadata[class_id] = {}

        if category is not None:
            self._metadata[class_id]["category"] = category
        if name_vi is not None:
            self._metadata[class_id]["name_vi"] = name_vi
        if name_en is not None:
            self._metadata[class_id]["name_en"] = name_en
        if standard_code is not None:
            self._metadata[class_id]["standard_code"] = standard_code

        self.save_classes()
        logger.info(f"Updated class ID {class_id} to '{clean_name}'")
        return self.format_item(class_id, clean_name, template)

    def delete_class(self, class_id: int) -> bool:
        if class_id not in self._classes:
            raise KeyError(f"Class ID {class_id} not found.")
        removed_name = self._classes.pop(class_id)
        self._metadata.pop(class_id, None)
        self.save_classes()
        logger.info(f"Deleted class ID {class_id}: '{removed_name}'")
        return True

    def reset_to_defaults(self, template: Optional[str] = None) -> ClassifierClassesResponse:
        self._classes = dict(DEFAULT_CLASSES)
        self._metadata = {k: dict(v) for k, v in DEFAULT_SIGN_METADATA.items()}
        self._categories = [dict(item) for item in DEFAULT_CATEGORIES]
        self.save_classes()
        logger.info("Reset classes configuration to default 100 classes with full metadata.")
        return self.get_all_classes_response(template)

    def get_prompts(self, template: Optional[str] = None) -> List[str]:
        active_template = template or config_manager.classifier_prompt_template or settings.CLASSIFIER_PROMPT_TEMPLATE
        prompts = []
        for cid in sorted(self._classes.keys()):
            raw_name = self._classes[cid]
            human_readable = raw_name.replace("_", " ")
            prompts.append(active_template.format(human_readable))
        return prompts


class_manager = ClassManager()
