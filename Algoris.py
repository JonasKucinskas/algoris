import sys
import ply.lex as lex
import ply.yacc as yacc


class Lexer():
    # List of token names.   This is always required
    
    # A string containing ignored characters (spaces and tabs)
    t_ignore  = ' \t'

    literals = {'=', '+', '-', '/', '*', '(', ')', '{', '}', ',', ';', '>', '<', '!'}
    
    #special keywords
    

    reserved = {
        'if' : 'IF',
        'else' : 'ELSE',
    }

    tokens = [
       'NUMBER',
       'PLUS',
       'MINUS',
       'TIMES',
       'DIVIDE',
       'LPAREN',
       'RPAREN',
       'EQ', 
       'NE',
       'LE',
       'ME',
       'INC',
    ] + list(reserved.values())

    # Regular expression rules for simple tokens
    t_PLUS    = r'\+'
    t_MINUS   = r'-'
    t_TIMES   = r'\*'
    t_DIVIDE  = r'/'
    t_LPAREN  = r'\('
    t_RPAREN  = r'\)'

    t_EQ = r'=='
    t_NE = r'!='
    t_LE = r'<='
    t_ME = r'>='
    t_INC = r'\+\+'

    
    #precedence = (
    #    ('left', '>', '<', t_LE, t_ME, t_EQ, t_NE),
    #    ('left', '!'),
    #    ('left', '+', '-'),
    #    ('left', '*', '/'),
    #    ('right', t_INC),
    #)

    # A regular expression rule with some action code
    
    def t_ID(t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        t.type = Lexer().reserved.get(t.value,'ID')  # Check for reserved words
        return t

    def t_NUMBER(t):
        r'\d+'
        t.value = int(t.value)    
        return t

    # Define a rule so we can track line numbers
    def t_newline(t):
        r'\n+'
        t.lexer.lineno += len(t.value)

    def t_CHAR(t):
        r'\'[^\']\''
        t.value = t.value[1]
        return t

    def t_STRING(t):
        r'\"[^\"]*\"'
        t.value = t.value.replace('"', '')
        return t

    # Error handling rule
    def t_error(t):
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)
        main()

    # Build the lexer
    lexer = lex.lex()    

class Parser():

    tokens = Lexer().tokens

    def p_expression_plus(p):
        'expression : expression PLUS term'
        p[0] = p[1] + p[3]

    def p_expression_minus(p):
        'expression : expression MINUS term'
        p[0] = p[1] - p[3]

    def p_expression_eq(p):
        'expression : factor EQ factor'
        p[0] = p[1] == p[3]

    def p_expression_ne(p):
        'expression : factor NE factor'
        p[0] = p[1] != p[3]

    def p_expression_inc(p):
        'expression : factor INC'
        p[0] = p[1] + 1

    def p_expression_le(p):
        'expression : factor LE factor'
        p[0] = p[1] <= p[3]

    def p_expression_me(p):
        'expression : factor ME factor'
        p[0] = p[1] >= p[3]

    def p_expression_term(p):
        'expression : term'
        p[0] = p[1]

    def p_term_times(p):
        'term : term TIMES factor'
        p[0] = p[1] * p[3]

    def p_term_div(p):
        'term : term DIVIDE factor'
        p[0] = p[1] / p[3]

    def p_term_factor(p):
        'term : factor'
        p[0] = p[1]

    def p_factor_num(p):
        'factor : NUMBER'
        p[0] = p[1]

    def p_factor_expr(p):
        'factor : LPAREN expression RPAREN'
        p[0] = p[2]

    # Error rule for syntax errors
    def p_error(p):
        print("Syntax error in input!")


 


    # Build the parser
    parser = yacc.yacc()

def main():
    lexer = Lexer().lexer
    parser = Parser().parser

    try:
        data = open(sys.argv[1]).read()
        
        if data:
            # Give the lexer input
            lexer.input(data)

            # Tokenize
            for tok in lexer:
                print(tok)
            result = parser.parse(data)
            print(result)
   

    except (IndexError, FileNotFoundError) as e:
        print(e)

if __name__ == '__main__':
    main()
