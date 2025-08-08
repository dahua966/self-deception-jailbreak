#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

API_HOST = ""
API_KEY = ""
DIRECT_KEY = ""
PUBLIC_MODEL = ""


MODEL_PATH = {"qwen3": f"{PUBLIC_MODEL}/Qwen3-8B",
              "deepseek": f"{PUBLIC_MODEL}/DeepSeek-R1-Distill-Llama-8B"}


PROJECT_DIR = str(Path(__file__).parent.parent.resolve())


Languages = {"zh":"Chinese", "en":"English", "ru":"Russian", "fr":"French", "es":"Spanish", "ar":"Arabic"}


# 对话场景
Senarios = {'en': ['game', 'opera', 'teleplay', 'script', 'novel', 'comic', 'movie'],
            'es': ['juego', 'ópera', 'televisión', 'guión', 'novela', 'cómic', 'película'],
            'zh': ['游戏', '歌剧', '电视剧', '剧本', '小说', '漫画', '电影'],
            'fr': ['jeu', 'opéra', 'télévision', 'scénario', 'roman', 'bande dessinée', 'film'],
            'ru': ['видеоигра', 'опера', 'телевидение', 'сценарий', 'роман', 'комикс', 'фильм'],
            'ar': ['لعبة', 'أوبرا', 'تلفزيون', 'سيناريو', 'رواية', 'كوميك', 'فيلم']}


sexual_adjective = {"en":['Obscene','Vulgar', 'Lewd','Indecent', 'Salacious', 'Raunchy', 'Steamy', 'licentious', 'lascivious', 'nasty'],
                    "zh": ['淫秽的', '好色的', '淫荡的', '肉欲的', '情色的', '下流的', '淫乱的', '性欲的', '荒淫的', '龌龊的'],
                    "fr": ['Obscène', 'Licencieux', 'Lubrique', 'Dépravé', 'Indécent',  'Vicieux', 'Goulu', 'méchant', 'débauché', 'déshonorant'],
                    "es": ['Obsceno', 'Lascivo', 'Libidinoso', 'Depravado', 'Indecente', 'Sensual', 'Vicioso', 'Caliente', 'Morboso', 'malvado'],
                    "ru": ['Обсценный', 'Грубый', 'Нагота', 'Непристойный','развратный', 'противный', 'Похабный', 'скандальный', 'некультурный', 'Вульгарный'],
                    "ar": ['فاضِح', 'نَذِل', 'سَافِل', 'قَذِر',  'بذيئة', 'فاحشة', 'قذر', 'فاضح', 'ملتوي' ,'مضلل']}

violence_adjective = {"en":['Fierce', 'Violent', 'Brutal', 'Savage','Aggressive', 'Bloody', 'Vicious', 'Ruthless', 'Barbaric', 'Merciless'], #'Ferocious', --> Merciless
                    "zh": ['凶狠的', '残忍的', '野蛮的', '暴虐的', '凶残的', '血腥的', '狂暴的', '强横的', '残暴的', '凶恶的'],
                    "fr": ['Violent', 'Brutal', 'Aggressif', 'Féroce', 'Sanglant', 'Barbare', 'Cruel', 'Inhumain', 'Impitoyable', 'meurtrier'], #'Sauvage', --> meurtrier
                    "es": ['Violento', 'Brutal', 'Agresivo', 'Sangriento', 'Bárbaro', 'Cruel', 'Inhumano', 'Implacable',  'tiránico', 'Ensangrentado'],
                    "ru": ['Сильный', 'Жестокий', 'Суровый', 'Агрессивный', 'Кровавый', 'Безжалостный', 'Отмороженный',  'порочный', 'тиранический', 'свирепый'],#дикий-->свирепый
                    "ar": ['عُنِيف', 'قاسٍ', 'ظالِم', 'مُتَوَحْش', 'شَرِّس', 'قاسي', 'طاغية', 'مُتَوَحَّش', 'شرير', 'عنيف']}

hate_adjective = {"en":['Prejudiced', 'Discriminatory', 'Hateful', 'Racist', 'Sexist', 'Homophobic', 'Xenophobic', 'Intolerant', 'Hostile', 'Antagonistic'],
                    "zh": ['偏见的',  '种族主义的', '性别主义的', '反同性恋的', '排外的', '仇视的', '民族主义的', '充满敌意的', '种姓主义的', '宗教极端主义的'],
                    "fr": ['Hostile', 'Rancunier', 'Malveillant', 'Antipathique', 'Méprisant', 'préjudiciable', 'Amer', 'Venimeux','Odieux', 'xénophobe'],
                    "es": ['Odioso', 'Hostil', 'Rencoroso', 'Malintencionado', 'Antipático', 'Despreciativo', 'xenófobo', 'prejuiciado', 'racista', 'sexista'],
                    "ru": ['Эксклюзивный', 'Враждебный', 'Злобный', 'Антипатичный', 'Презрительный', 'Ненавистный', 'Ядовитый', 'Ужасный', 'расистский', 'сексистский'],
                    "ar": ['عَدُوّ', 'عَدُوّي', 'مُبْغِض', 'بَغِيض', 'مُتَعَصِّب', 'كراهية' , 'عنصري', "متطرف ديني", "معادٍ للمثليين", 'بُغض']}

init_vectors = {"violence": violence_adjective, "hate": hate_adjective, "sexual": sexual_adjective}