VAR a = []
VAR b = []
FOR i = 0 TO 100 THEN
    APPEND(a, i)
END

VAR count = COUNT(a)

WHILE count < 100000 THEN
    PRINT(count)
    STARTTEST(count)
        FOR j = 0 TO count THEN
            APPEND(b, j)
        END
    STOPTEST()
    VAR a = a*a
    VAR count = COUNT(a)
END
MAKEGRAPH()