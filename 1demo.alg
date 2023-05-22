VAR a = []
VAR b = []
FOR i = 0 TO 100 THEN
    APPEND(a, i)
END

VAR count = COUNT(a)

WHILE count < 9999 THEN
    PRINT(count)
    STARTTEST(count)
        PRINT("a")
    STOPTEST()
    VAR a = a+1
    VAR count = COUNT(a)
END
MAKEGRAPH()