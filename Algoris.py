import sys
import ply.lex as lex
import ply.yacc as yacc


class Lexer():
    # List of token names.   This is always required
    
    # A string containing ignored characters (spaces and tabs)
    t_ignore  = ' \t'

    literals = {'=', '+', '-', '/', '*', '(', ')', '{', '}', ',', ';', '>', '<', '!'}


    tokens = (
       'NUMBER',
       'PLUS',
       'MINUS',
       'TIMES',
       'DIVIDE',
       'LPAREN',
       'RPAREN',
    )

    # Regular expression rules for simple tokens
    t_PLUS    = r'\+'
    t_MINUS   = r'-'
    t_TIMES   = r'\*'
    t_DIVIDE  = r'/'
    t_LPAREN  = r'\('
    t_RPAREN  = r'\)'

    # A regular expression rule with some action code
    def t_NUMBER(t):
        r'\d+'
        t.value = int(t.value)    
        return t

    # Define a rule so we can track line numbers
    def t_newline(t):
        r'\n+'
        t.lexer.lineno += len(t.value)

    def CHAR(self, t):
        r'\'[^\']\''
        t.value = t.value[1]
        return t

    def STRING(self, t):
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
    # Test it out

class Parser():

    tokens = Lexer().tokens

    def p_expression_plus(p):
        'expression : expression PLUS term'
        p[0] = p[1] + p[3]

    def p_expression_minus(p):
        'expression : expression MINUS term'
        p[0] = p[1] - p[3]

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
        s = input('calc > ')
        if data:

            # Give the lexer input
            lexer.input(data)

            # Tokenize
            for tok in lexer:
                print(tok)

            while True:
                try:
                    s = input('calc > ')
                except EOFError:
                    break
                if not s: continue
                result = parser.parse(s)
                print(result)
   

    except (IndexError, FileNotFoundError) as e:
        print(e)

if __name__ == '__main__':
    main()
