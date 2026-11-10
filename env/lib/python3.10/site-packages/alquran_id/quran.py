#! /usr/bin/env python

import xml.etree.ElementTree as et
from os import path as p
from alquran_id.surat import *

d = p.dirname(p.abspath(__file__))
quran = p.join(d, "source/quran.xml")
terjemahan = p.join(d, "source/terjemahan.xml")
tafsir = p.join(d, "source/tafsir.xml")


class AlQuran():


    def __init__(self):
        self.quran = et.parse(quran)
        self.terjemahan = et.parse(terjemahan)
        self.tafsir = et.parse(tafsir)

    def ArNumber(self, number):
        number = str(number)
        ar = {
            '0': '۰',
            '1': '۱',
            '2': '۲',
            '3': '۳',
            '4': '۴',
            '5': '۵',
            '6': '۶',
            '7': '۷',
            '8': '۸',
            '9': '۹',
            }
        return "".join([ar[char] for char in number])

    def Surat(self, id_surat):
        arabic_surat = self.quran.getroot()
        arabic_surat = arabic_surat.find(f".//sura[@index='{id_surat}']")
        arabic_surat = arabic_surat.get("name")
        nama_surat = list(idSurat)[int(id_surat)-1]
        return arabic_surat, nama_surat

    def Ayat(self, id_surat, id_ayat):
        surat = self.quran.getroot()
        surat = surat.find(f".//sura[@index='{id_surat}']")
        ayat = surat.find(f".//aya[@index='{id_ayat}']")
        ayat = ayat.get("text")
        return ayat

    def JumlahAyat(self, id_surat):
        surat = self.quran.getroot()
        surat = surat.find(f".//sura[@index='{id_surat}']")
        jml_ayat = surat.findall("aya")
        jml_ayat = len(jml_ayat)
        return jml_ayat

    def Terjemahan(self, id_surat, id_ayat):
        terjemahan = self.terjemahan.getroot()
        terjemahan = terjemahan.find(f".//sura[@index='{id_surat}']")
        ayat = terjemahan.find(f".//aya[@index='{id_ayat}']")
        ayat = ayat.get("text")
        return ayat

    def Tafsir(self, id_surat, id_ayat):
        tafsir = self.tafsir.getroot()
        tafsir = tafsir.find(f".//sura[@index='{id_surat}']")
        ayat = tafsir.find(f".//aya[@index='{id_ayat}']")
        ayat = ayat.get("text")
        return ayat
