from src.contraponto.claims import Alegacao
from src.contraponto.search import Evidencia
from src.contraponto.verifier import VerificadorAlegacoes

verificador = VerificadorAlegacoes()

# Teste 1: Alegação sem evidências -> Deve gerar abstenção (evidencia_insuficiente)
alegacao_1 = Alegacao(id=1, texto="Um meteoro invisível atingiu a lua ontem.")
veredito_1 = verificador.verificar_alegacao(alegacao_1, [])

print("--- TESTE 1: SEM EVIDÊNCIAS (ABSTENÇÃO CONSERVADORA) ---")
print(f"Alegação: {veredito_1.texto_alegacao}")
print(f"Categoria: {veredito_1.categoria.upper()}")
print(f"Justificativa: {veredito_1.justificativa}\n")

# Teste 2: Alegação com evidência de refutação
alegacao_2 = Alegacao(
    id=2, texto="Vacina da covid foi banida em todos os países"
)
evidencias_2 = [
    Evidencia(
        titulo="Vacina contra covid não foi banida",
        url="https://www.estadao.com.br/estadao-verifica/brasil-vacina-covid-banida-outros-paises/",
        publicador="Estadão Verifica",
        trecho="Veredito do Checador: Enganoso",
        tipo_fonte="fact-check",
    )
]

veredito_2 = verificador.verificar_alegacao(alegacao_2, evidencias_2)

print("--- TESTE 2: COM EVIDÊNCIA DE FACT-CHECK ---")
print(f"Alegação: {veredito_2.texto_alegacao}")
print(f"Categoria: {veredito_2.categoria.upper()}")
print(f"Justificativa: {veredito_2.justificativa}")
print(f"Fontes: {veredito_2.fontes_citadas}")