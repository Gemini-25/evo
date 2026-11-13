6.2.0 바탕 단계의 공용 시험 자료 (스포츠 설계 4.1 「바탕(0)」, D28·D44).

묶음 P(presets/*)가 엔진의 형식 4 부품(밀기 Move, 균형 보조 Balance: 묶음 R)과 종목 몸 틀(biped·disc: 묶음 PM)을 기다리지 않고
틀 읽기·펼치기·넣기·조리법 만들기를 개발하고 시험하게 하려는 것이다. 몸 틀 둘은 형식 1 부품만 쓴다.

  fixture_walker.evotpl  몸 틀(옆에서 본 세계): 몸통 + 두 다리, 회전 관절 2, 모터 2(키 Q·A / W·S), 몸 상태·관절 각도·접촉 센서.
                         슬라이더 size·power, 빈자리 $ground(태그 ground → terrain 차례: D44), 랜덤 값 tilt와 시작 자세 수식(설계 3.1의
                         Template.randoms·Template.spawn).
  fixture_puck.evotpl    몸 틀(위에서 본 세계): 원 하나 + 추진기 넷(방향키), 몸 상태 센서. 물체의 바람 항력(drag, 형식 1)으로 멈춘다.
  fixture_side.json      조리법(옆): 경기장 track_side(묶음 P의 일반 틀) + fixture_walker, 규칙 distance·touch_ends·time_limit, train.
                         만든 결과는 형식 1이어야 한다(얼린 v6.1이 여는 파일: T26).
  fixture_top.json       조리법(위): 경기장 yard_top(묶음 P의 일반 틀) + fixture_puck, 규칙 distance·save_energy·time_limit, brain·evaluation.
                         만든 결과는 형식 4다(세계 종류 view = top 때문. 부품은 형식 1).

모양의 근거: 틀 = 부분 설계 DU 2.2(format·version·id·kind·world·name·description·params·root·parts·slots) + 최종 설계 3.1(randoms, spawn,
slots.<이름>.tags). 사용자 틀처럼 글을 직접 담는다(name·label: D55. 내장 틀만 name_id·label_id). 쓰지 않는 칸(keys·evolvable·spawns·rules)은
적지 않았다. 조리법 = DU 2.3 + 최종 설계 D28(brain·evaluation·train)·D41(players.mirror 없음, 팀 0 자리만). 규칙 칸 이름은 strings.py
RULE_SENTENCES의 자리 이름(part·direction·per_m·what·outcome·seconds·weight)과 DU 2.3의 who(역할 id).

가정(묶음 P가 정하는 것과 맞아야 한다. 다르면 통합 담당에게 알려 이 파일을 고친다):
  - 경기장 틀 track_side·yard_top에 자리 이름 "start"가 있다(players.seats[].spawn).
  - track_side의 땅은 id "ground"·태그 "ground"(설계 3.1), 빈자리 $ground가 그것으로 풀린다.
  - 조리법 규칙의 "what": "$ground"는 그 역할 몸 틀의 빈자리 이름이다(DU 2.3의 보기 그대로).

tests/test_foundation_v62.py가 이 파일들의 모양(JSON, 틀의 수식이 기본·최소·최대 슬라이더 값에서 환경 모델에 맞게 펼쳐짐, 형식 1 부품만)을 지킨다.
