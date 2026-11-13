"""「새 프리셋」 창과 틀 보관함 창 (6.2.0, 최종 설계 3.1 W·D42, 설계 DU 4.1·4.2·5.1).

- wizard.py: PresetWizard(다섯 단계 안내 창), open_wizard.
- pages.py: 다섯 단계의 쪽.
- cards.py: 틀 카드·슬라이더 위젯·보관함 창(TemplatePicker, 에디터 팔레트 「틀」 단추가 함께 씀).
- preview.py: 미리보기(지금 조리법으로 만든 환경을 그리고, 행동 0으로 몇 초 돌려 보임).

화면 글은 모두 strings.py에 있다(이 꾸러미에 한글 글자 상수 없음, T38).
"""
