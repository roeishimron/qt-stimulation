import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from stims import inflate_randomley
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText, DEFAULT_FONT
from itertools import cycle
from viewing_experiment import ViewExperiment
from random import choices, shuffle, randint, choice

COMMON_HEBREW_WORDS = ("שלנו,מאוד,אותו,יותר,הרבה,ואני,לאחר,היתה,המים,החוף,אותם,שהוא,אחרי,לפני,שאני,האלה,כאשר,כמעט,בעוד,גדול,יכול,שהיה,נראה,אותי,תמיד,במשך,רבים,לתוך"
                       ",צריך,סביב,אולי,אותה,במים,מקום,טובה,בתוך,יהיה,עצמי,שאתה,לחוף,באמת,בחור,לומר,אליו,בטוח,רואה,חושב,דקות,מאשר,איזה,עליו,הכול,הזאת,אותך,ירוק,אגוז"
                       ",השקט,בזמן,משהו,ואכן,לעבר,קרוב,היער,מתוך,קודם,עצים,אומר,קקאו,מתחת,כאלה,למדי,השמש,ולכן,זוכר,ללכת,נכון,גבוה,הקשת,שהיא,רוצה,ברגע,ובכן,מבין,מלבד,למים,היום,העמק,הקטן"
                       ",רחוק,חייב,שלהם,מוזר,הדבר,שאין,פחות,איתו,במבט,בדרך,לדבר,יודע,פניו,במצב,השני,השטח,הרוח,איים,הניח,בהיר,בקצה,בלתי,שהיו,עצמו,פגעה,העין,אליה,היטב,פיסת,אפשר,בבית,הערב,אחרת"
                       ",מהרה,לדעת,שנים,חברי,והוא,והיה,החלק,המסע,דגים,ארוך,סירה,קטנה,רבות,בקול,השיב,חיים,מעבר,ראשו,לגבי,עושה,כתפו,הפרי,האדם,בחוף,סיבה,איזו,הארץ,קושי,ונתן,ביום,למטה,נשבר,נוכל,נדמה,מבעד,המזל"
                       ",ממנה,חיוך,כחול,וכמה,תודה,הטוב,וחצי,זינק,ובכל,בוקר,העיר,צורה,הגיע,לקבל,עבור,חומר,צעיר,הרחב,לילה,שחור,הבית,רובר,טבעי,רצון,אילו,היקר,נותן,ימים,בפעם,בנים,פנים,מהיר,סכנה,נושא"
                       ",נעים,כמות,מפני,מדבר,לקחת,עולה,היית,תהיה,עליה,בידי,הזמן,החפץ,שכבר,כוחו,מעמד,ממנו,בשפע,נעשה,אותן,לשים,הדרך,רגיל,ואתה,ואמר"
                       ",שעוד,עליז,וחתך,עלים,הללו,ענקי,נמצא,מראה,מוכן,הגזע,שלוש,מגיע,עשרה,ושוב,רחצה,ביער,נקרא,חנית,תפוח,חזיר,הספר,חזהו,משני,הלכה,ההוא,נעשו,וככל"
                       ",אלפי,יפים,מענג,למעט,דעתי,ילדי,קפטן"
                       ).split(",")


def scramble_str(s: str) -> str:
    mutable = list(s)
    shuffle(mutable)
    # add a final letter in the middle of the word for clarity
    mutable[randint(0, len(mutable) - 2)] = choice("ץךףןם")
    return "".join(mutable)


SCRAMBELED = [scramble_str(w) for w in COMMON_HEBREW_WORDS]


def create_appliable_text(t: str) -> AppliableText:
    # is_different_font = choices([True, False], [0.1, 0.9])[0]
    # if is_different_font:
    #     return AppliableText(t, font_family="Yehuda CLM", font_size=50)
    return AppliableText(t, font_size=50)


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    size = int(screen_height*3/4)

    stimuli = map(create_appliable_text,
                  inflate_randomley(COMMON_HEBREW_WORDS, 10))
    oddballs = map(create_appliable_text, inflate_randomley(SCRAMBELED, 10))

    main_window = ViewExperiment(OddballStimuli(
        size, cycle(stimuli), cycle(oddballs), 3), SoftSerial(), 3, trial_duration=45)
    main_window.show()

    # Run the main Qt loop
    app.exec()
