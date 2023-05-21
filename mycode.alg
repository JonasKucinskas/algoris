VAR a = []
VAR b = []
FOR i = 0 TO 100 THEN
    APPEND(a, i)
END

VAR count = COUNT(a)

WHILE count < 1600 THEN
    PRINT(count)
    STARTTEST(count)
    FOR i = 0 TO count THEN 	
        FOR j = 0 TO count THEN
            APPEND(b, j)
        END
    END
    STOPTEST()
    VAR a = DOUBLE(a)
    VAR count = COUNT(a)
END
MAKEGRAPH()