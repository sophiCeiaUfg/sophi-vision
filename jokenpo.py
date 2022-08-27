ganha = {
    'Pedra': 'Tesoura', 'Tesoura': 'Papel', 'Papel': 'Pedra'
}

def whoWins(user, pc):
    if user == pc:
      return 'EMPATE!'
    elif ganha[user] == pc:
      return 'VOCE GANHOU!!1!'
    elif ganha[pc] == user:
      return 'VOCE PERDEU...'