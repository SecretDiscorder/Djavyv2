#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from .jvdict import Jvdict

def ganti(inp, index, character):
    return inp[0:int(index)] + character

def ganti2(inp, index, character):
    return inp[0:index-1] + character

def ganti3(inp, index, character):
    return inp[0:index-2] + character

def transliterate(content, regexp_file):
    trans = content
    content_split = list(content)
    j = 0
    for i in range(0, len(content_split)):
        if "ꦂ" in regexp_file and regexp_file["ꦂ"] == "r":  # jawa->latin
            if content_split[i] == "ꦲ":  # ha
                if i > 0 and (content_split[i-1] == "ꦼ" or content_split[i-1] == "ꦺ" or content_split[i-1] == "ꦶ" or content_split[i-1] == "ꦴ" or content_split[i-1] == "ꦸ" or content_split[i-1] == "ꦄ" or content_split[i-1] == "ꦌ" or content_split[i-1] == "ꦆ" or content_split[i-1] == "ꦎ" or content_split[i-1] == "ꦈ"):
                    trans = ganti(trans, j, "h"+regexp_file[content_split[i]])
                    j += 2
                if i > 0 and (content_split[i-1] == "꧊"):
                    trans = ganti(trans, j, "H"+regexp_file[content_split[i]])
                    j += 2
                else:
                    trans = ganti(trans, j, regexp_file.get(content_split[i], content_split[i]))
                    j += 1
            elif i > 0 and content_split[i] == "ꦫ" and content_split[i-1] == "ꦂ":  # double rr
                trans = ganti(trans, j, "a")
                j += 1
            elif i > 0 and content_split[i] == "ꦔ" and content_split[i-1] == "ꦁ":  # double ngng
                trans = ganti(trans, j, "a")
                j += 1
            elif content_split[i] == "ꦴ" or content_split[i] == "ꦶ" or content_split[i] == "ꦸ" or content_split[i] == "ꦺ" or content_split[i] == "ꦼ":
                if i > 2 and content_split[i-1] == "ꦲ" and content_split[i-2] == "ꦲ":  # -hah-
                    if content_split[i] == "ꦴ":
                        trans = ganti3(trans, j, "ā")
                    elif content_split[i] == "ꦶ":
                        trans = ganti3(trans, j, "ai")
                    elif (content_split[i] == "ꦸ"):
                        trans = ganti3(trans, j, "au")
                    elif (content_split[i] == "ꦺ"):
                        trans = ganti3(trans, j, "ae")
                    elif (content_split[i] == "ꦼ"):
                        trans = ganti3(trans, j, "aě")
                elif i > 2 and content_split[i-1] == "ꦲ":  # -h-
                    if (content_split[i] == "ꦴ"):
                        trans = ganti3(trans, j, "ā")
                    elif (content_split[i] == "ꦶ"):
                        trans = ganti3(trans, j, "i")
                    elif (content_split[i] == "ꦸ"):
                        trans = ganti3(trans, j, "u")
                    elif (content_split[i] == "ꦺ"):
                        trans = ganti3(trans, j, "e")
                    elif (content_split[i] == "ꦼ"):
                        trans = ganti3(trans, j, "ě")
                    j -= 1
                elif (i > 0 and content_split[i] == "ꦴ" and content_split[i-1] == "ꦺ"):  # -o #2 aksara -> 1 huruf
                    trans = ganti2(trans, j, "o")
                elif (i > 0 and content_split[i] == "ꦴ" and content_split[i-1] == "ꦻ"):  # -au #2 aksara -> 2 huruf
                    trans = ganti3(trans, j, "au")
                elif (content_split[i] == "ꦴ"):  # -aa
                    trans = ganti(trans, j, "aa")
                    j += 1
                elif i > 0 and (content_split[i] == "ꦶ" or content_split[i] == "ꦸ" or content_split[i] == "ꦺ" or content_split[i] == "ꦼ") and (content_split[i-1] == "ꦄ" or content_split[i-1] == "ꦌ" or content_split[i-1] == "ꦆ" or content_split[i-1] == "ꦎ" or content_split[i-1] == "ꦈ"):
                    trans = ganti(trans, j, regexp_file[content_split[i]])
                    j += 1
                else:
                    trans = ganti2(trans, j, regexp_file[content_split[i]])
            elif content_split[i] == "ꦽ" or content_split[i] == "ꦾ" or content_split[i] == "ꦿ" or content_split[i] == "ꦷ" or content_split[i] == "ꦹ" or content_split[i] == "ꦻ" or content_split[i] == "ꦇ" or content_split[i] == "ꦍ":  # 1 aksara -> 2 huruf
                trans = ganti2(trans, j, regexp_file[content_split[i]])
                j += 1
            elif content_split[i] == "꦳":  # 2 aksara -> 2 huruf
                if i > 0 and content_split[i-1] == "ꦗ":
                    if i > 1 and content_split[i-2] == "꧊":
                        trans = ganti3(trans, j, "Za")
                    else:
                        trans = ganti3(trans, j, "za")
                elif i > 0 and content_split[i-1] == "ꦥ":
                    if i > 1 and content_split[i-2] == "꧊":
                        trans = ganti3(trans, j, "Fa")
                    else:
                        trans = ganti3(trans, j, "fa")
                elif i > 0 and content_split[i-1] == "ꦮ":
                    if i > 1 and content_split[i-2] == "꧊":
                        trans = ganti3(trans, j, "Va")
                    else:
                        trans = ganti3(trans, j, "va")
                else:
                    trans = ganti2(trans, j, regexp_file[content_split[i]])
            elif content_split[i] == "꧀":
                trans = ganti2(trans, j, regexp_file[content_split[i]])
            elif i > 1 and content_split[i] == "ꦕ" and content_split[i-1] == "꧀" and content_split[i-2] == "ꦚ":  # nyj & nyc
                trans = ganti2(trans, j-2, "nc")
            elif i > 1 and content_split[i] == "ꦗ" and content_split[i-1] == "꧀" and content_split[i-2] == "ꦚ":  # nyj & nyc
                trans = ganti2(trans, j-2, "nj")
            elif content_split[i] == "ꦏ" or content_split[i] == "ꦐ" or content_split[i] == "ꦑ" or content_split[i] == "ꦒ" or content_split[i] == "ꦓ" or content_split[i] == "ꦕ" or content_split[i] == "ꦖ" or content_split[i] == "ꦗ" or content_split[i] == "ꦙ" or content_split[i] == "ꦟ" or content_split[i] == "ꦠ" or content_split[i] == "ꦡ" or content_split[i] == "ꦢ" or content_split[i] == "ꦣ" or content_split[i] == "ꦤ" or content_split[i] == "ꦥ" or content_split[i] == "ꦦ" or content_split[i] == "ꦧ" or content_split[i] == "ꦨ" or content_split[i] == "ꦩ" or content_split[i] == "ꦪ" or content_split[i] == "ꦫ" or content_split[i] == "ꦬ" or content_split[i] == "ꦭ" or content_split[i] == "ꦮ" or content_split[i] == "ꦯ" or content_split[i] == "ꦱ" or content_split[i] == "ꦉ" or content_split[i] == "ꦊ" or content_split[i] == "ꦁ":
                if i > 0 and content_split[i-1] == "꧊":
                    if content_split[i] == "ꦐ":
                        trans = ganti(trans, j, "Qa")
                        j += 2
                    elif content_split[i] == "ꦧ" or content_split[i] == "ꦨ":
                        trans = ganti(trans, j, "Ba")
                        j += 2
                    elif content_split[i] == "ꦕ" or content_split[i] == "ꦖ":
                        trans = ganti(trans, j, "Ca")
                        j += 2
                    elif content_split[i] == "ꦢ" or content_split[i] == "ꦣ":
                        trans = ganti(trans, j, "Da")
                        j += 2
                    elif content_split[i] == "ꦒ" or content_split[i] == "ꦓ":
                        trans = ganti(trans, j, "Ga")
                        j += 2
                    elif content_split[i] == "ꦗ" or content_split[i] == "ꦙ":
                        trans = ganti(trans, j, "Ja")
                        j += 2
                    elif content_split[i] == "ꦏ" or content_split[i] == "ꦑ":
                        trans = ganti(trans, j, "Ka")
                        j += 2
                    elif content_split[i] == "ꦭ":
                        trans = ganti(trans, j, "La")
                        j += 2
                    elif content_split[i] == "ꦩ":
                        trans = ganti(trans, j, "Ma")
                        j += 2
                    elif content_split[i] == "ꦤ" or content_split[i] == "ꦟ":
                        trans = ganti(trans, j, "Na")
                        j += 2
                    elif content_split[i] == "ꦥ" or content_split[i] == "ꦦ":
                        trans = ganti(trans, j, "Pa")
                        j += 2
                    elif content_split[i] == "ꦫ" or content_split[i] == "ꦬ":
                        trans = ganti(trans, j, "Ra")
                        j += 2
                    elif content_split[i] == "ꦱ" or content_split[i] == "ꦯ":
                        trans = ganti(trans, j, "Sa")
                        j += 2
                    elif content_split[i] == "ꦠ" or content_split[i] == "ꦡ":
                        trans = ganti(trans, j, "Ta")
                        j += 2
                    elif content_split[i] == "ꦮ":
                        trans = ganti(trans, j, "Wa")
                        j += 2
                    elif content_split[i] == "ꦪ":
                        trans = ganti(trans, j, "Ya")
                        j += 2
                    else:
                        ganti(trans, j, regexp_file[content_split[i]])
                        j += 3
                elif content_split[i] == "ꦑ" or content_split[i] == "ꦓ" or content_split[i] == "ꦖ" or content_split[i] == "ꦙ" or content_split[i] == "ꦡ" or content_split[i] == "ꦣ" or content_split[i] == "ꦦ" or content_split[i] == "ꦨ" or content_split[i] == "ꦯ":  # bha, cha, dha, dll.
                    trans = ganti(trans, j, regexp_file[content_split[i]])
                    j += 3
                else:  # ba, ca, da, dll.
                    trans = ganti(trans, j, regexp_file[content_split[i]])
                    j += 2
            elif content_split[i] == "ꦰ":  # ṣa
                trans = ganti(trans, j, regexp_file[content_split[i]])
                j += 2
            elif content_split[i] == "ꦔ" or content_split[i] == "ꦘ" or content_split[i] == "ꦚ" or content_split[i] == "ꦛ" or content_split[i] == "ꦜ" or content_split[i] == "ꦝ" or content_split[i] == "ꦞ" or content_split[i] == "ꦋ":
                if i > 0 and content_split[i-1] == "꧊":
                    if content_split[i] == "ꦔ":
                        trans = ganti(trans, j, "Nga")
                        j += 3
                    elif content_split[i] == "ꦚ" or content_split[i] == "ꦘ":
                        trans = ganti(trans, j, "Nya")
                        j += 3
                    elif content_split[i] == "ꦛ" or content_split[i] == "ꦜ":
                        trans = ganti(trans, j, "Tha")
                        j += 3
                    elif content_split[i] == "ꦝ" or content_split[i] == "ꦞ":
                        trans = ganti(trans, j, "Dha")
                        j += 3
                    else:
                        ganti(trans, j, regexp_file[content_split[i]])
                        j += 3
                else:
                    trans = ganti(trans, j, regexp_file[content_split[i]])
                    j += 3
            elif content_split[i] == "꧊":  # penanda nama diri -- made up for Latin back-compat
                trans = ganti(trans, j, "")
            elif content_split[i] == " ":
                trans = ganti(trans, j, " ")
                j += 1
            else:
                trans = ganti(trans, j, regexp_file.get(content_split[i], content_split[i]))
                j += 1
        elif "r" in regexp_file and regexp_file["r"] == "ꦂ":  # latin->jawa
            if content_split[i] == "a" and i > 0:
                trans = ganti(trans, j, " ")
                j += 1
            else:
                trans = ganti(trans, j, regexp_file.get(content_split[i], content_split[i]))
                j += 1
        else:
            trans = ganti(trans, j, content_split[i])
            j += 1

    return trans

def print_help():
    print("""
    This is a program for transliterating text in Aksara Jawa to Latin script. Please provide the text you want to see transliterated as a single argument to this script (say, put it in quotation marks on the command line).
    """)

dicts = Jvdict()

if len(sys.argv) == 1:
    print_help()
else:
    if len(sys.argv) == 2:
        to_translate = sys.argv[1]
    else:
        to_translate = " ".join(sys.argv[1:])

    print(transliterate(to_translate, dicts.return_javtolatin()))
