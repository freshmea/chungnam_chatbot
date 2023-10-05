# tic tac toe game monte carlo tree search
def visual_board(board_num: dict) -> None:
    print(board_num["1"], "|", board_num["2"], "|", board_num["3"])
    print("--+--+--")
    print(board_num["4"], "|", board_num["5"], "|", board_num["6"])
    print("--+--+--")
    print(board_num["7"], "|", board_num["8"], "|", board_num["9"])


def game(boarder):
    turn = "X"
    count = 0
    for i in range(8):
        visual_board(boarder)
        print("당신의 차례입니다." + turn + "어디로 이동할까요?")
        move = input()
        if boarder[move] == " ":
            boarder[move] = turn
            count += 1
        else:
            print("이미 이동되었습니다. 다른 곳을 선택하세요.")
            continue

        if count >= 5:
            if boarder["1"] == boarder["2"] == boarder["3"] != " ":
                visual_board(boarder)
                print("\n게임 끝!" + turn + "가 이겼습니다.")
                break
            elif boarder["4"] == boarder["5"] == boarder["6"] != " ":
                visual_board(boarder)
                print("\n게임 끝!" + turn + "가 이겼습니다.")
                break
            elif boarder["7"] == boarder["8"] == boarder["9"] != " ":
                visual_board(boarder)
                print("\n게임 끝!" + turn + "가 이겼습니다.")
                break
            elif boarder["1"] == boarder["4"] == boarder["7"] != " ":
                visual_board(boarder)
                print("\n게임 끝!" + turn + "가 이겼습니다.")
                break
            elif boarder["2"] == boarder["5"] == boarder["8"] != " ":
                visual_board(boarder)
                print("\n게임 끝!" + turn + "가 이겼습니다.")
                break
            elif boarder["3"] == boarder["6"] == boarder["9"] != " ":
                visual_board(boarder)
                print("\n게임 끝!" + turn + "가 이겼습니다.")
                break
            elif boarder["7"] == boarder["5"] == boarder["3"] != " ":
                visual_board(boarder)
                print("\n게임 끝!" + turn + "가 이겼습니다.")
                break
            elif boarder["1"] == boarder["5"] == boarder["9"] != " ":
                visual_board(boarder)
                print("\n게임 끝!" + turn + "가 이겼습니다.")
                break
        if count == 9:
            print("\n게임 종료.\n")
            print("비겼습니다!")

        if turn == "X":
            turn = "O"
        else:
            turn = "X"


def main():
    boarder = dict()
    for i in range(9):
        boarder[str(i + 1)] = " "
    board_keys = []
    for key in boarder:
        board_keys.append(key)
    game(boarder)


if __name__ == "__main__":
    main()
