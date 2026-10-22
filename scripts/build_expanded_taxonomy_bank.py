"""Expanded Grounded VLM Taxonomy Query Bank (32 Rich Queries).

Covers:
1. Textual KIS: 12 ultra-specific target queries with fine-grained visual cues.
2. Temporal Alignment: 8 sequential multi-event queries with logical time progression.
3. Ad-hoc Video Search (AVS): 12 broad multi-positive concepts across 695 videos.
Both in English (EN) and Vietnamese (VI).
"""
import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

EXPANDED_BANK = {
    "textual_kis": [
        {
            "id": "KIS_01",
            "theme": "Phòng sạch Bán dẫn",
            "target_video_id": "O34mNBDyYPE",
            "query_en": "Laboratory technician in white cleanroom bunny suit inspecting silicon semiconductor wafer under yellow lighting",
            "query_vi": "Kỹ thuật viên phòng thí nghiệm mặc trang phục phòng sạch màu trắng kiểm tra đĩa bán dẫn silicon dưới ánh đèn vàng",
            "discriminative_cues": "White bunny suit, silicon wafer, yellow cleanroom lighting"
        },
        {
            "id": "KIS_02",
            "theme": "Thảm họa Cháy Phà Biển",
            "target_video_id": "K-VSAqEjTHI",
            "query_en": "Passenger ferry on fire billowing thick black smoke while passengers in orange life jackets board inflatable rescue boat",
            "query_vi": "Phà chở khách bốc cháy nghi ngút khói đen trong khi các hành khách mặc áo phao màu cam lên thuyền cứu hộ",
            "discriminative_cues": "Burning ferry, black smoke, orange life jackets on rescue boat"
        },
        {
            "id": "KIS_03",
            "theme": "Âu Tàu Đập Tam Hiệp & Pin Mặt Trời",
            "target_video_id": "qG97ZIHYSfY",
            "query_en": "Giant concrete ship lock canal of Three Gorges Dam with massive blue photovoltaic solar panels on surrounding hills",
            "query_vi": "Âu tàu bê tông khổng lồ của đập Tam Hiệp với các tấm pin năng lượng mặt trời màu xanh trên sườn đồi xung quanh",
            "discriminative_cues": "Three Gorges concrete canal, massive water lock, blue solar panels"
        },
        {
            "id": "KIS_04",
            "theme": "Cứu hộ Đổ nát Động đất",
            "target_video_id": "y9BigFu0d7k",
            "query_en": "Emergency search and rescue workers in yellow helmets digging through cracked concrete rubble with trained search dog",
            "query_vi": "Lực lượng cứu hộ đội mũ bảo hộ màu vàng đang đào bới đống đổ nát bê tông nứt vỡ cùng chó nghiệp vụ",
            "discriminative_cues": "Yellow helmets, cracked concrete debris, search dog"
        },
        {
            "id": "KIS_05",
            "theme": "Giải Bóng rổ Làng quê (Thôn BA)",
            "target_video_id": "7LF0Vva1j54",
            "query_en": "Outdoor basketball court in rural village surrounded by thousands of cheering spectators holding banners under floodlights",
            "query_vi": "Sân bóng rổ ngoài trời tại làng quê bao quanh bởi hàng ngàn khán giả cổ vũ cầm biểu ngữ dưới ánh đèn pha",
            "discriminative_cues": "Outdoor basketball game, packed cheering rural crowd, floodlights"
        },
        {
            "id": "KIS_06",
            "theme": "Tấn công Tàu Hàng Biển Đỏ",
            "target_video_id": "xppaKLnZPmA",
            "query_en": "Commercial cargo vessel damaged by projectile strike smoking on ocean with naval warship escorting nearby",
            "query_vi": "Tàu chở hàng thương mại bị bốc khói trên đại dương sau vụ tấn công với tàu chiến hộ tống gần đó",
            "discriminative_cues": "Cargo container vessel, smoke from deck, naval patrol boat"
        },
        {
            "id": "KIS_07",
            "theme": "Nhật thực Toàn phần & Lưới Điện",
            "target_video_id": "E",
            "query_en": "Total solar eclipse corona in dark sky above high voltage electricity transmission pylons and solar power field",
            "query_vi": "Vành nhật hoa của nhật thực toàn phần trên bầu trời tối phía trên các cột điện cao thế và cánh đồng pin mặt trời",
            "discriminative_cues": "Solar eclipse corona, high voltage pylons, dark midday sky"
        },
        {
            "id": "KIS_08",
            "theme": "Tái thiết Sau Cháy rừng Lahaina",
            "target_video_id": "LDQRXWGiFSA",
            "query_en": "Aerial view of burned foundation ruins and newly constructed wooden houses in coastal town rebuilding after wildfire",
            "query_vi": "Góc nhìn từ trên cao về tàn tích móng nhà bị cháy và những ngôi nhà gỗ mới xây tại thị trấn ven biển sau vụ cháy rừng",
            "discriminative_cues": "Burned coastal neighborhood, foundation footprints, rebuilding construction"
        },
        {
            "id": "KIS_09",
            "theme": "Tỷ giá Đồng Yên & Can thiệp Ngân hàng",
            "target_video_id": "5uJ0ZdS-zcM",
            "query_en": "Financial trading room displays showing Japanese Yen currency exchange rate charts and Bank of Japan headquarters building",
            "query_vi": "Màn hình phòng giao dịch tài chính hiển thị biểu đồ tỷ giá đồng Yên Nhật và tòa nhà trụ sở Ngân hàng Trung ương Nhật Bản",
            "discriminative_cues": "Currency rate candlestick chart, Tokyo financial district, Bank of Japan facade"
        },
        {
            "id": "KIS_10",
            "theme": "Thuyền Người Di cư Cập bến",
            "target_video_id": "Trs049E",
            "query_en": "Overcrowded black rubber dinghy carrying migrants wearing orange life jackets being intercepted by border patrol cutter",
            "query_vi": "Xuồng cao su màu đen chở người di cư quá tải mặc áo phao cam bị tàu tuần tra biên giới tiếp cận",
            "discriminative_cues": "Inflatable black boat, orange life jackets, border force vessel"
        },
        {
            "id": "KIS_11",
            "theme": "Kiểm soát Vũ khí Thái Lan",
            "target_video_id": "dEk-32GMbkY",
            "query_en": "Thai police officers in uniform displaying confiscated handguns and rifles on official press conference table",
            "query_vi": "Sĩ quan cảnh sát Thái Lan mặc quân phục trưng bày súng ngắn và súng trường bị tịch thu trên bàn họp báo",
            "discriminative_cues": "Thai police emblem, confiscated firearms on table, microphones"
        },
        {
            "id": "KIS_12",
            "theme": "Lễ Tang Cựu Thủ tướng",
            "target_video_id": "jnMTmrR3aTM",
            "query_en": "Official state memorial hall draped in black and yellow ribbons with portrait of former statesman surrounded by white flowers",
            "query_vi": "Hội trường tưởng niệm quốc gia trang trí dải ruy băng đen vàng với di ảnh cựu lãnh đạo bao quanh bởi hoa trắng",
            "discriminative_cues": "State memorial portrait, white floral wreaths, black ribbon drapery"
        }
    ],

    "temporal_alignment": [
        {
            "id": "TEMP_01",
            "theme": "Cháy rừng Tiến sát -> Đoàn xe Sơ tán",
            "video_id_context": "aIA6RE5e6_c",
            "steps_en": [
                "Intense orange fire glow and thick smoke approaching tree line along highway",
                "Fire trucks and police convoy escorting civilian cars evacuating down highway"
            ],
            "steps_vi": [
                "Ánh lửa đỏ rực và khói dày đặc tiến sát rặng cây dọc theo đường cao tốc",
                "Đoàn xe cứu hỏa và xe cảnh sát hộ tống người dân sơ tán trên đường cao tốc"
            ],
            "temporal_logic": "Lửa tiếp cận đường -> Xe cộ sơ tán có cảnh sát hộ tống (t1 < t2)"
        },
        {
            "id": "TEMP_02",
            "theme": "Cấp cứu Hồi sức -> Bệnh nhân Tỉnh táo",
            "video_id_context": "8iZSkgqoI",
            "steps_en": [
                "Doctors and intensive care nurses monitoring patient with digital ECG heart rate displays",
                "Patient sitting up talking with doctor holding medical clipboard in hospital ward"
            ],
            "steps_vi": [
                "Bác sĩ và y tá hồi sức cấp cứu theo dõi bệnh nhân với màn hình điện tâm đồ ECG",
                "Bệnh nhân ngồi dậy nói chuyện với bác sĩ cầm tập hồ sơ bệnh án trong phòng bệnh"
            ],
            "temporal_logic": "Theo dõi cấp cứu hồi sức -> Bệnh nhân tỉnh táo phục hồi (t1 < t2)"
        },
        {
            "id": "TEMP_03",
            "theme": "Bão Đổ bộ Sóng lớn -> Cây đổ Ngập đường",
            "video_id_context": "IX6GDdiKafU",
            "steps_en": [
                "Violent ocean waves and storm surge crashing against concrete sea wall in heavy rain",
                "Uprooted trees and fallen street billboards blocking flooded city street after typhoon"
            ],
            "steps_vi": [
                "Sóng biển dữ dội và triều cường đập mạnh vào bờ kè chắn sóng bê tông trong mưa lớn",
                "Cây cối bật gốc và biển quảng cáo ngã đổ chắn ngang con đường ngập nước sau bão"
            ],
            "temporal_logic": "Sóng bão đập bờ kè -> Cây đổ ngập đường sau bão (t1 < t2)"
        },
        {
            "id": "TEMP_04",
            "theme": "Tên lửa trên Bệ -> Khởi động Bay lên",
            "video_id_context": "OpXlGrKUPmo",
            "steps_en": [
                "Rocket standing vertical on steel launch structure against blue sky",
                "Massive smoke plume and exhaust flame as rocket lifts off launch pad"
            ],
            "steps_vi": [
                "Tên lửa đứng thẳng trên giàn phóng bằng thép dưới bầu trời xanh",
                "Cột khói khổng lồ và ngọn lửa phụt ra khi tên lửa rời bệ phóng"
            ],
            "temporal_logic": "Tên lửa đứng yên trên giàn -> Động cơ kích hoạt bay lên (t1 < t2)"
        },
        {
            "id": "TEMP_05",
            "theme": "Hạn hán Cạn kiệt -> Cứu trợ Nước sạch",
            "video_id_context": "w",
            "steps_en": [
                "Cracked parched reservoir bed with dried mud and stranded wooden boats",
                "Municipal water tanker truck pumping fresh water into plastic barrels for waiting residents"
            ],
            "steps_vi": [
                "Lòng hồ khô cạn nứt nẻ với bùn khô và thuyền gỗ mắc cạn",
                "Xe bồn chở nước của thành phố bơm nước sạch vào thùng nhựa cho người dân đang xếp hàng"
            ],
            "temporal_logic": "Hồ chứa cạn kiệt -> Xe bồn đến phân phát nước (t1 < t2)"
        },
        {
            "id": "TEMP_06",
            "theme": "Cảnh sát Đột kích -> Áp giải Tòa án",
            "video_id_context": "dEk-32GMbkY",
            "steps_en": [
                "Heavily armed tactical police convoy arriving outside residence building with flashing sirens",
                "Suspect in handcuffs escorted by plainclothes detectives down courthouse staircase"
            ],
            "steps_vi": [
                "Đoàn xe cảnh sát đặc nhiệm có vũ trang tiến đến trước tòa nhà với còi hụ chớp sáng",
                "Nghi phạm bị còng tay được các trinh sát cảnh sát áp giải bước xuống cầu thang tòa án"
            ],
            "temporal_logic": "Cảnh sát bao vây đột kích -> Áp giải nghi phạm ra tòa (t1 < t2)"
        },
        {
            "id": "TEMP_07",
            "theme": "Sóng Thần Động đất -> Đội Cứu trợ Dựng Lều",
            "video_id_context": "y9BigFu0d7k",
            "steps_en": [
                "Seismic shaking damages multi-story concrete buildings leaving rubble on streets",
                "Military soldiers and volunteers erecting blue humanitarian emergency tents on soccer field"
            ],
            "steps_vi": [
                "Rung chấn địa chấn làm hư hại các tòa nhà bê tông để lại đống đổ nát trên đường phố",
                "Binh lính quân đội và tình nguyện viên dựng các lều nhân đạo khẩn cấp màu xanh trên sân bóng"
            ],
            "temporal_logic": "Động đất gây đổ nát -> Lực lượng cứu nạn dựng lều sơ tán (t1 < t2)"
        },
        {
            "id": "TEMP_08",
            "theme": "Đám đông Tụ tập -> Khán đài Ăn mừng",
            "video_id_context": "7LF0Vva1j54",
            "steps_en": [
                "Villagers walking in large groups carrying flags and drums entering outdoor sports stadium",
                "Spectators jumping and waving red scarves celebrating winning goal with colorful fireworks"
            ],
            "steps_vi": [
                "Người dân làng đi thành từng nhóm lớn mang theo cờ và trống tiến vào sân vận động thể thao",
                "Khán giả nhảy múa và vẫy khăn quàng đỏ ăn mừng bàn thắng chiến thắng cùng pháo hoa rực rỡ"
            ],
            "temporal_logic": "Khán giả tiến vào sân -> Cả khán đài ăn mừng chiến thắng (t1 < t2)"
        }
    ],

    "adhoc_video_search": [
        {
            "id": "AVS_01",
            "concept": "Hạ tầng Năng lượng Sạch Tái tạo",
            "query_en": "Solar photovoltaic power panels and wind turbine generators in rural landscapes",
            "query_vi": "Các tấm pin năng lượng mặt trời và tuabin gió trên các vùng cảnh quan nông thôn",
            "target_multi_videos": "qG97ZIHYSfY, xBOG9k57w_E, wuQ26k-Lnh4"
        },
        {
            "id": "AVS_02",
            "concept": "Hạn hán Nghiêm trọng & Thiếu Nước sạch",
            "query_en": "Parched cracked dry soil reservoirs and citizens carrying plastic water containers",
            "query_vi": "Hồ chứa nước khô cạn nứt nẻ và người dân mang can nhựa đi lấy nước",
            "target_multi_videos": "o8L5ZXBBdjs, w, V6XAeLdeI"
        },
        {
            "id": "AVS_03",
            "concept": "An ninh Hàng hải & Hoạt động Tàu chiến",
            "query_en": "Commercial cargo container ships and naval warships operating in open ocean waters",
            "query_vi": "Tàu chở hàng container thương mại và tàu chiến hải quân hoạt động trên vùng biển mở",
            "target_multi_videos": "xppaKLnZPmA, K-VSAqEjTHI, 47EBP1i1XLc"
        },
        {
            "id": "AVS_04",
            "concept": "Xe điện & Dây chuyền Sản xuất Ô tô",
            "query_en": "Modern electric vehicles and automobile assembly line production in manufacturing factory",
            "query_vi": "Xe điện hiện đại và dây chuyền lắp ráp sản xuất ô tô trong nhà máy",
            "target_multi_videos": "fYgYEoeoYS4, wuQ26k-Lnh4, jK7ueKiAYxU"
        },
        {
            "id": "AVS_05",
            "concept": "Cứu trợ Khẩn cấp Sau Động đất",
            "query_en": "First responders setting up emergency triage tents and distributing relief supplies to victims",
            "query_vi": "Lực lượng phản ứng nhanh dựng lều cứu trợ khẩn cấp và phân phát nhu yếu phẩm cho nạn nhân",
            "target_multi_videos": "y9BigFu0d7k, Wbm09_gVdqU, TCqAlw1uTQc"
        },
        {
            "id": "AVS_06",
            "concept": "Tập kích Tên lửa & Phòng không Đô thị",
            "query_en": "Anti-aircraft air defense missiles launching at night sky and smoke over city skyline",
            "query_vi": "Tên lửa phòng không bắn lên bầu trời đêm và khói bốc lên trên đường chân trời thành phố",
            "target_multi_videos": "OpXlGrKUPmo, 2lS7aGGw, DH5GqQwCY0A"
        },
        {
            "id": "AVS_07",
            "concept": "Khủng hoảng Di cư Băng biển Địa Trung Hải",
            "query_en": "Migrants in life jackets arriving on sandy beach rescued by humanitarian volunteers",
            "query_vi": "Người di cư mặc áo phao lên bãi biển cát được tình nguyện viên nhân đạo hỗ trợ",
            "target_multi_videos": "47EBP1i1XLc, Trs049E, AtDXBWMbJtI"
        },
        {
            "id": "AVS_08",
            "concept": "Nắng nóng Khắc nghiệt & Cháy rừng Châu Âu",
            "query_en": "Wildfire flames burning pine forest and thermometer showing high temperature heatwave",
            "query_vi": "Ngọn lửa cháy rừng thiêu rụi rừng thông và nhiệt kế hiển thị nhiệt độ cao trong đợt nắng nóng",
            "target_multi_videos": "wZnyt7wSlRg, o8L5ZXBBdjs, aIA6RE5e6_c"
        },
        {
            "id": "AVS_09",
            "concept": "Phòng mổ & Thiết bị Y tế Kỹ thuật cao",
            "query_en": "Surgeons wearing sterile blue surgical scrubs performing operation under bright operating lights",
            "query_vi": "Các bác sĩ phẫu thuật mặc áo mổ màu xanh thực hiện ca phẫu thuật dưới ánh đèn phòng mổ",
            "target_multi_videos": "8iZSkgqoI, mRcU6eSqRko, V6XAeLdeI"
        },
        {
            "id": "AVS_10",
            "concept": "Bục Họp báo & Phát biểu Chính khách",
            "query_en": "Government spokesperson delivering official speech behind wooden podium with microphones",
            "query_vi": "Người phát ngôn chính phủ phát biểu chính thức đằng sau bục gỗ có gắn micro",
            "target_multi_videos": "4c1P1hiBSAc, j09oMrfkvlM, hfoK8ZUyQJQ"
        },
        {
            "id": "AVS_11",
            "concept": "Lễ hội Truyền thống & Trình diễn Nghệ thuật",
            "query_en": "Dancers performing acrobatic choreography with colorful traditional costumes and stage lighting",
            "query_vi": "Các vũ công biểu diễn vũ đạo nhào lộn với trang phục truyền thống đầy màu sắc và ánh sáng sân khấu",
            "target_multi_videos": "NcVbh5WZ1T4, LzI5l1ZVL8g, 7LF0Vva1j54"
        },
        {
            "id": "AVS_12",
            "concept": "Bão Lũ & Ngập úng Giao thông Đô thị",
            "query_en": "High floodwaters submerging city vehicles and residents wading through muddy street currents",
            "query_vi": "Nước lũ dâng cao nhấn chìm các phương tiện trong thành phố và người dân lội qua dòng nước bùn",
            "target_multi_videos": "IX6GDdiKafU, sMtzQk1V79U, RhfcZzt9fD8"
        }
    ]
}

out_file = ROOT_DIR / "storage" / "newsseek" / "vlm_grounded_queries_benchmark.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(EXPANDED_BANK, f, indent=2, ensure_ascii=False)

print(f"Successfully generated expanded taxonomy bank:")
print(f"- Textual KIS: {len(EXPANDED_BANK['textual_kis'])} queries")
print(f"- Temporal: {len(EXPANDED_BANK['temporal_alignment'])} sequences")
print(f"- AVS: {len(EXPANDED_BANK['adhoc_video_search'])} concepts")
print(f"Total: {len(EXPANDED_BANK['textual_kis']) + len(EXPANDED_BANK['temporal_alignment']) + len(EXPANDED_BANK['adhoc_video_search'])} rich queries")
print(f"Saved to: {out_file}")
