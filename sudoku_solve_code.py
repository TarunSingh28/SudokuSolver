

def solve_try(arr):
    count=0
    for i in range(9):
        for j in range(9):
            if arr[i][j]==0:
                x=i
                y=j
                count=1
                break
        if count:
            break
    if not count:
        return True
    else:
        for t in range(1,10):
            good=1
            for i in range(9):
                if arr[i][y]==t or arr[x][i]==t:
                    good=0
                    break
            if good:
                ii=3*(x//3)
                jj=3*(y//3)
                for i in range(3):
                    for j in range(3):
                        if arr[i+ii][j+jj]==t:
                            good=0
                            break
            if good:
                arr[x][y]=t
                if solve_try(arr):
                    return True
                else:
                    arr[x][y]=0
        return False
    
