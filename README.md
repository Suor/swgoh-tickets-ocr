# Распознавалка скринов купонов для SWGOH

Делает то, что заявлено) Понимает ники на русском и английском, но легко добавить и другие языки.
Требуется регистрация всех на [swgoh.gg](https://swgoh.gg/) или ручное составление списка ников.


## Установка

Требуется Python 3, tesseract и ещё кое-что. Инструкция по установке идёт для Ubuntu:

```bash
# Ставим гит, питон и тессеракт
sudo apt-get install git python3 python3-venv tesseract-ocr-rus tesseract-ocr-eng

# Подтягиваем код
git clone https://github.com/Suor/swgoh-tickets-ocr.git
cd swgoh-tickets-ocr

# Создаём виртуальное окружение и ставим зависимости
python3.6 -m venv venv
source ./venv/bin/activate
pip install -r requerements.txt

# Конфигурируем приложение
echo '{"guild_url": "https://swgoh.gg/g/your/guildurl/"}' > config.json
python update_dict.py  # Или вручную пишем ники в dict.txt по одному на строку
```


## Использование

```sh
# Если виртуальное окружение активировано
python tickets.py screenshot.png screenshot2.png

# Если ничего не активировано, например, из другой программы, крона и т.п.
/path/to/venv/bin/python tickets.py screenshot.png screenshot2.png

# Чтобы переподтянуть список ников
/path/to/venv/bin/python update_dict.py
```

Результат выдаётся в виде JSON:

```js
{
    'tickets': {
        'Mr Hey': 440,
        'jo3': 300,
        'Барка': 268,
        'Hackflow': 600
    },
    'warnings': ['No count for Ерёма'],
    'total': 27868
}
```
