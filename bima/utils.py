import numpy as np

def check_winner(board):
    # Periksa baris
    for row in board:
        if len(set(row)) == 1 and row[0] != "":
            return row[0]
    
    # Periksa kolom
    for col in board.T:
        if len(set(col)) == 1 and col[0] != "":
            return col[0]
    
    # Periksa diagonal utama
    if len(set(board.diagonal())) == 1 and board[0, 0] != "":
        return board[0, 0]
    
    # Periksa diagonal sekunder
    if len(set(np.fliplr(board).diagonal())) == 1 and board[0, 2] != "":
        return board[0, 2]
    
    return None

def check_draw(board):
    return not np.any(board == "")

