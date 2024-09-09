from ast import keyword
import sys, os
hruf_latin = {
    "a" : "._",
    "b" : "_...",
    "c" : "_._.",
    "d" : "_..",
    "e" : ".",
    "f" : ".._.",
    "g" : "__.",
    "h" : "....",
    "i" : "..",
    "j" : ".___",
    "k" : "_._",
    "l" : "._..",
    "m" : "__",
    "n" : "_.",
    "o" : "___",
    "p" : ".__.",
    "q" : "__._",
    "r" : "._.",
    "s" : "...",
    "t" : "_",
    "u" : "..._",
    "w" : ".__",
    "x" : "_.._",
    "y" : "_.__",
    "z" : "__..",
    "1" : ".._",
    "v" : ".____",
    "2" : "..___",
    "3" : "...__",
    "4" : "...._",
    "5" : ".....",
    "6" : "_....",
    "7" : "__...",
    "8" : "___..",
    "9" : "____.",
    "0" : "_____"
}

def morse_translate(text):
    morse_code = []
    for char in text.lower():
        if char in hruf_latin:
            morse_code.append(hruf_latin[char])
        else:
            morse_code.append(" ")  # Menggunakan spasi jika karakter tidak ada dalam dictionary
    return " ".join(morse_code)

def reverse_morse_translate(morse_code):
    reverse_dict = {value: key for key, value in hruf_latin.items()}
    morse_chars = morse_code.strip().split(" ")
    text = "".join([reverse_dict[char] if char in reverse_dict else " " for char in morse_chars])
    return text

