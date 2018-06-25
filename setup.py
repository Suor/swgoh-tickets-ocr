from setuptools import setup

setup(
    name='swgoh-tickets-ocr',
    version='0.1',
    author='Alexander Schepanovski',
    author_email='suor.web@gmail.com',

    description='Parse SWGoH tickets screenshots.',
    long_description=open('README.md').read(),
    url='http://github.com/Suor/swgoh-tickets-ocr',
    license='BSD',

    py_modules=['tickets', 'update_dict'],
    install_requires=[
        'funcy==1.10.2',
        # OCR
        'opencv-python==3.4.1.15',
        'Pillow==5.1.0',
        'pytesseract==0.2.2',
        # Scraping
        'cssselect==1.0.3',
        'lxml==4.2.1',
        'requests==2.19.1',
        'python-Levenshtein==0.12.0',
    ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',

        'Intended Audience :: Developers',
        'Topic :: Games/Entertainment',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)
