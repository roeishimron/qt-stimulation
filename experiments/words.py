COMMON_HEBREW_WORDS = "היה,אבל,אני,הוא,לנו,היו,אמר,כמו,כדי,אתה,שלי,פרק,אחד,טוב,זאת,כפי,הזה,הים,זמן,האי,כמה,קרא,בין,בכל,שלו,אלה,מעל,שבו,שכן,אשר,פעם,ידי,אחר,קטן,ולא,מכל,ואת,שלא,שזה,העץ,מהם,כאן,ואז,פני,מכן,שלך,שוב,אחת,שום,דבר,אין,בלי,עוד,בים,קצת,מיד,היא,ללא,שני,אלא,למה,ספק,בעל,דרך,הנה,ענה,מזה,לים,רבה,סלע,שגק,קצר,יחד,שהם,שעה,רגע,כלל,יום,תוך,שלה,שיש,מדי,שאל,עשה,האם,צעק,כבר,לקח,חצי,ליד,ההר,לכל,אכן,מים,איך,בהם,מטר,לאט,וזה,עצי,עלה,חוץ,עמק,וכך,יפה,בוא,מעט,רוח,אלי,נוח,ממש,ראש,אבן,לבש,ושם,בכך,ענף,אמי,ועל,סוף,אנו,כזה,תפס,להם,זהו,וגם,חור,חתך,זרק,ילד,פרי,ועד,ככל,שבה,מזג,עלי,איי,לתת,קרה,נפל,וגק,שכב,שפת,קטע,עשו,הון,הכי,כלי,חלק,חזר,בשל,הלך,קשה,אדם,האש,הקו,הלב,אור,אבי,איש,קשר,רוב,זקן,ערב,לזה,שקט,החץ,הלו,פתח,שאם,נתן,צעד,הבא,וכל,כוח,דחף,הגב,זוג,לאי,מאז,לבן,נהג,מזל,חדש,החל,בזה,חוט,משם,מין,הדג,גרם,מול,והם,קול,שתי,הנר,עבר,שמו,רבע,מהן,ראה,תחת,חמש,פנו,הצד,באי,מצב,פיו,בשם,הבר,סוג,להב,הכל,בהן,קשת,פחד,ומה,אמת,טרח,הפה,יתר,עצר,רעב,בשר,ורץ,עבה,ידע,יבש,קצה,גדם,חול,צבע,נגע,קלה,ואם,חום,חפץ,מבט,לעץ,צדו,בגן,חזק,הבד,עתה,פרץ,סתם,מצד,גבר,ארץ,צער".split(",")

COMMON_ENGLISH_WORDS =  [
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "had", "her", "was", "one", "our", "out", "day", "get", "has", "him",
    "his", "how", "man", "new", "now", "old", "see", "two", "way", "who",
    "boy", "did", "its", "let", "put", "say", "she", "too", "use", "act",
    "add", "age", "ago", "aim", "air", "art", "ask", "bad", "bag", "bar",
    "bat", "bed", "bee", "bet", "big", "bit", "box", "bus", "bye", "cap",
    "car", "cat", "cow", "cry", "cup", "cut", "dad", "die", "dig", "dog",
    "dot", "dry", "due", "ear", "eat", "egg", "end", "era", "eye", "fan",
    "far", "fat", "fee", "few", "fit", "fix", "fly", "fog", "fun", "fur",
    "gap", "gas", "gem", "got", "guy", "gym", "hat", "hit", "hot", "ice",
    "ill", "ink", "ion", "jam", "jet", "job", "joy", "key", "kid", "kit",
    "lab", "lad", "lap", "law", "lay", "leg", "lid", "lip", "log", "low",
    "mad", "map", "mat", "may", "met", "mix", "mom", "mud", "net", "nod",
    "nor", "nut", "odd", "off", "oil", "own", "pad", "pan", "pay", "pen",
    "pet", "pie", "pig", "pin", "pot", "raw", "ray", "red", "rid", "row",
    "rub", "rug", "run", "sad", "sea", "set", "sew", "sir", "sit", "sky",
    "son", "sun", "tag", "tap", "tax", "tea", "ten", "tie", "tip", "toe",
    "top", "toy", "try", "unit", "up", "van", "via", "war", "wet", "win",
    "yes", "yet", "zip", "zoo"
]

ARABIC_LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
HEBREW_LETTERS = "אבגדהוזחטיכלמנסעפצקרשתףךץםן"

def into_arabic(s: str) -> str:
    return "".join((ARABIC_LETTERS[((ord(c)-ord("א"))%(len(ARABIC_LETTERS)))] for c in s))

def into_digits(s: str) -> str:
    AMOUNT_OF_LETTERS = 27
    return "".join((str(int((ord(c)-ord("א"))/AMOUNT_OF_LETTERS*10)) for c in s))