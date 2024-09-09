#!/usr/bin/python3

# -*- coding: utf-8 -*-

class Jvdict():

    def __init__(self):

        self.javtolatin = {
        "ꦀ":'', #? -- archaic
        "ꦁ":'ng', #cecak
        "ꦂ":'r', #layar
        "ꦃ":'h', #wignyan
        "ꦄ":'A', #swara-A
        "ꦅ":'I', #I-Kawi -- archaic
        "ꦆ":'I', #I
        "ꦇ":'Ii', #Ii -- archaic
        "ꦈ":'U', #U
        "ꦉ":'rě', #pa cêrêk
        "ꦊ":'lě', #nga lêlêt
        "ꦋ":'lěu', #nga lêlêt Raswadi -- archaic
        "ꦌ":'E', #E
        "ꦍ":'Ai', #Ai
        "ꦎ":'O', #O

        "ꦏ":'ka',
        "ꦐ":'qa', #Ka Sasak
        "ꦑ":'kha', #Murda
        "ꦒ":'ga',
        "ꦓ":'gha', #Murda
        "ꦔ":'nga',
        "ꦕ":'ca',
        "ꦖ":'cha', #Murda
        "ꦗ":'ja',
        "ꦘ":'Nya', #Ja Sasak, Nya Murda
        "ꦙ":'jha', #Ja Mahaprana
        "ꦚ":'nya',
        "ꦛ":'tha', #'ṭa',
        "ꦜ":'ṭha', #Murda
        "ꦝ":'dha', #'ḍa',
        "ꦞ":'ḍha', #Murda
        "ꦟ":'ṇa', #Murda
        "ꦠ":'ta',
        "ꦡ":'tha', #Murda
        "ꦢ":'da',
        "ꦣ":'dha', #Murda
        "ꦤ":'na',
        "ꦥ":'pa',
        "ꦦ":'pha', #Murda
        "ꦧ":'ba',
        "ꦨ":'bha', #Murda
        "ꦩ":'ma',
        "ꦪ":'ya',
        "ꦫ":'ra',
        "ꦬ":'Ra', #Ra Agung
        "ꦭ":'la',
        "ꦮ":'wa',
        "ꦯ":'sha', #Murda
        "ꦰ":'ṣa', #Sa Mahaprana
        "ꦱ":'sa',
        "ꦲ":'a', #could also be "a" or any sandhangan swara

        "꦳":'​', #cecak telu -- diganti zero-width joiner (tmp)
        "ꦺꦴ":'o', #taling tarung
        "ꦴ":'a',
        "ꦶ":'i',
        "ꦷ":'ii',
        "ꦸ":'u',
        "ꦹ":'uu',
        "ꦺ":'e',
        "ꦻ":'ai',
        "ꦼ":'ě',
        "ꦽ":'rě',
        "ꦾ":'ya',
        "ꦿ":'ra',

        "꧀":'​', #pangkon -- diganti zero-width joiner (tmp)

        "꧁":'—',
        "꧂":'—',
        "꧃":'–',
        "꧄":'–',
        "꧅":'–',
        "꧆":'',
        "꧇":'​', #titik dua -- diganti zero-width joiner (tmp)
        "꧈":',',
        "꧉":'.',
        "꧊":'qqq',
        "꧋":'–',
        "꧌":'–',
        "꧍":'–',
        "ꧏ":'²',
        "꧐":'0',
        "꧑":'1',
        "꧒":'2',
        "꧓":'3',
        "꧔":'4',
        "꧕":'5',
        "꧖":'6',
        "꧗":'7',
        "꧘":'8',
        "꧙":'9',
        "꧞":'—',
        "꧟":'—',
        "​":'#', #zero-width joiner
        "​":' ' #zero-width space
        }

        self.latintojav = {
         #"":'ꦀ', #? -- archaic
        "ng":'ꦁ', #cecak
        "r":'ꦂ', #layar
        "h":'ꦃ', #wignyan
        "a":'ꦄ', #swara-A
        "i":'ꦅ', #I-Kawi -- archaic
        "i":'ꦆ', #I
        "ii":'ꦇ', #Ii -- archaic
        "u":'ꦈ', #U
        "rê":'ꦉ', #pa cêrêk
        "rě":'ꦉ', #pa cěrěk
        "lê":'ꦊ', #nga lêlêt
        "lě":'ꦊ', #nga lělět
        "lêu":'ꦋ', #nga lêlêt Raswadi -- archaic
        "lěu":'ꦋ', #nga lělět Raswadi -- archaic
        "e":'ꦌ', #E
        "ai":'ꦍ', #Ai
        "o":'ꦎ', #O

        "ka":'ꦏ',
        "qa":'ꦐ', #Ka Sasak
        "kha":'ꦑ', #Murda
        "ga":'ꦒ',
        "gha":'ꦓ', #Murda
        "nga":'ꦔ',
        "ca":'ꦕ',
        "cha":'ꦖ', #Murda
        "ja":'ꦗ',
        "Nya":'ꦘ', #Ja Sasak, Nya Murda
        "jha":'ꦙ', #Ja Mahaprana
        "nya":'ꦚ',
        "ṭa":'ꦛ',
        "ṭha":'ꦜ', #Murda
        "ḍa":'ꦝ',
        "ḍha":'ꦞ', #Murda
        "ṇa":'ꦟ', #Murda
        "ta":'ꦠ',
        "tha":'ꦡ', #Murda
        "da":'ꦢ',
        "dha":'ꦣ', #Murda
        "na":'ꦤ',
        "pa":'ꦥ',
        "pha":'ꦦ', #Murda
        "ba":'ꦧ',
        "bha":'ꦨ', #Murda
        "ma":'ꦩ',
        "ya":'ꦪ',
        "ra":'ꦫ',
        "Ra":'ꦬ', #Ra Agung
        "la":'ꦭ',
        "wa":'ꦮ',
        "sha":'ꦯ', #Murda
        "ṣa":'ꦰ', #Sa Mahaprana
        "sa":'ꦱ',
        "a":'ꦲ', #could also be "a" or any sandhangan swara

        "​":'꦳', #cecak telu -- diganti zero-width joiner (tmp)
        "o":'ꦺꦴ', #taling tarung
        "aa":'ꦴ',
        "i":'ꦶ',
        "ii":'ꦷ',
        "u":'ꦸ',
        "uu":'ꦹ',
        "e":'ꦺ',
        "ai":'ꦻ',
        "ê":'ꦼ',
        "rê":'ꦽ',
        "ě":'ꦼ',
        "rě":'ꦽ',
        "ya":'ꦾ',
        "ra":'ꦿ',

        "​":'꧀', #pangkon -- diganti zero-width joiner (tmp)

        "":'꧁',
        "":'꧂',
        "":'꧃',
        "":'꧄',
        "":'꧅',
        "":'꧆',
        "":'꧇', #/titik dua
        ",":'꧈',
        ".":'꧉',
        "":'꧊',
        "":'꧋',
        "(":'꧌',
        ")":'꧍',
         #"":'ꧏ',
        "0":'꧐',
        "1":'꧑',
        "2":'꧒',
        "3":'꧓',
        "4":'꧔',
        "5":'꧕',
        "6":'꧖',
        "7":'꧗',
        "8":'꧘',
        "9":'꧙',
        "":'꧞',
        "":'꧟',
        "#":'​', #zero-width joiner
        " ":'​' #zero-width space
        }

    def return_javtolatin(self):
        return (self.javtolatin)

    def return_latintojav(self):
        return (self.latintojav)

if __name__ == "__main__":
    print ("This is a dictionary file not intended for direct use.\nIt contains transliterations of letters in Jawanese script to latin script and vice versa.")
