"""Sessões curtas de diálogo socrático, isoladas por usuário e chat."""

import json
import re
import unicodedata
from urllib.parse import urlparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from llm import ProvedorLLM, criar_provedor_llm
from verifier import VereditoAlegacao


@dataclass
class EstadoSocratico:
    """Estado mínimo e temporário de uma conversa socrática."""

    noticia_original: str
    vereditos: list[VereditoAlegacao]
    pergunta_atual_index: int = 0
    respostas_usuario: list[str] = field(default_factory=list)
    perguntas_geradas: list[str] = field(default_factory=list)
    relacoes_turnos: list[str] = field(default_factory=list)
    expira_em: datetime = field(default_factory=lambda: datetime.now() + timedelta(minutes=30))


class GerenciadorSocratico:
    """Faz até três perguntas sem transformar respostas do usuário em fatos."""

    MAX_PERGUNTAS = 3

    def __init__(
        self, llm: ProvedorLLM | None = None, ttl_minutos: int = 30,
    ) -> None:
        self.llm = llm if llm is not None else criar_provedor_llm()
        self.ttl = timedelta(minutes=ttl_minutos)
        self._sessoes: dict[tuple[int, int], EstadoSocratico] = {}

    @staticmethod
    def _chave(user_id: int, chat_id: int | None) -> tuple[int, int]:
        return (chat_id if chat_id is not None else user_id, user_id)

    def _limpar_expiradas(self) -> None:
        agora = datetime.now()
        for chave in [k for k, estado in self._sessoes.items() if estado.expira_em <= agora]:
            del self._sessoes[chave]

    def iniciar_sessao(
        self,
        user_id: int,
        noticia_original: str,
        vereditos: list[VereditoAlegacao],
        persona_modo: str = "Acolhedor",
        chat_id: int | None = None,
    ) -> str:
        """Inicia ou substitui uma sessão no par chat/usuário."""
        del persona_modo  # Mantido somente por compatibilidade com a API anterior.
        self._limpar_expiradas()
        estado = EstadoSocratico(
            noticia_original=noticia_original[:4000],
            vereditos=list(vereditos),
            expira_em=datetime.now() + self.ttl,
        )
        pergunta = (
            "Ao comparar a notícia com o que foi encontrado, o que mais chamou sua atenção "
            "ou gerou desconfiança?"
        )
        estado.perguntas_geradas.append(pergunta)
        self._sessoes[self._chave(user_id, chat_id)] = estado
        return (
            "🏛️ Diálogo Socrático\n"
            "Vamos pensar juntos sobre a notícia. Vou responder ao que você disser usando "
            "o que foi encontrado na checagem, sem tratar sua opinião como um fato. "
            "Você também pode fazer perguntas e usar /cancelar quando quiser.\n\n"
            f"{self._resumo_investigacao(estado)}\n\n"
            f"Pergunta 1/3: {pergunta}"
        )

    def processar_resposta(
        self, user_id: int, resposta_texto: str, chat_id: int | None = None,
    ) -> str:
        """Registra uma reflexão e avança sem considerá-la evidência factual."""
        self._limpar_expiradas()
        chave = self._chave(user_id, chat_id)
        estado = self._sessoes.get(chave)
        if estado is None:
            return "Não há diálogo socrático ativo. Inicie pelo botão de um relatório."

        resposta = " ".join(resposta_texto.strip().split())[:500]
        if not resposta:
            return "Não consegui identificar sua resposta. Pode escrevê-la de outra forma?"

        ultimo_turno = estado.pergunta_atual_index + 1 >= self.MAX_PERGUNTAS
        turno = self._analisar_turno(estado, resposta, ultimo_turno)
        estado.respostas_usuario.append(resposta)
        estado.relacoes_turnos.append(turno["relacao"])
        estado.pergunta_atual_index += 1
        estado.expira_em = datetime.now() + self.ttl
        if ultimo_turno:
            return self._gerar_sintese_final(chave, turno["resposta"])

        pergunta = turno["pergunta"]
        estado.perguntas_geradas.append(pergunta)
        numero = estado.pergunta_atual_index + 1
        return f'{turno["resposta"]}\n\nPergunta {numero}/3: {pergunta}'

    def cancelar_sessao(self, user_id: int, chat_id: int | None = None) -> str:
        self._limpar_expiradas()
        if self._sessoes.pop(self._chave(user_id, chat_id), None) is not None:
            return "🚪 Diálogo socrático encerrado."
        return "Nenhum diálogo socrático ativo para encerrar."

    def tem_sessao_ativa(self, user_id: int, chat_id: int | None = None) -> bool:
        self._limpar_expiradas()
        return self._chave(user_id, chat_id) in self._sessoes

    @staticmethod
    def _contexto(estado: EstadoSocratico) -> dict:
        return {
            "alegacoes_verificadas": [
                {
                    "id": v.alegacao_id,
                    "alegacao": GerenciadorSocratico._texto_curto(v.texto_alegacao, 280),
                    "resultado": v.categoria,
                    "explicacao_validada": GerenciadorSocratico._explicacao_simples(v),
                    "fontes": list(dict.fromkeys(
                        urlparse(url).netloc for url in v.fontes_citadas if urlparse(url).netloc
                    )),
                }
                for v in estado.vereditos
            ],
            "historico_de_reflexoes_nao_verificadas": estado.respostas_usuario[-2:],
        }

    @staticmethod
    def _texto_curto(texto: str, limite: int) -> str:
        limpo = " ".join(texto.split())
        if len(limpo) <= limite:
            return limpo
        return limpo[: limite - 1].rsplit(" ", 1)[0] + "…"

    @staticmethod
    def _trecho_encontrado(justificativa: str) -> str:
        correspondencia = re.search(r'[“\"]([^“\"]{20,})[”\"]', justificativa)
        if not correspondencia:
            return ""
        trecho = " ".join(correspondencia.group(1).split())
        if len(trecho) > 180:
            trecho = trecho[:177].rsplit(" ", 1)[0] + "…"
        return trecho

    @classmethod
    def _explicacao_simples(cls, veredito: VereditoAlegacao | None) -> str:
        if veredito is None:
            return "Ainda não há informação suficiente para avaliar essa afirmação."
        explicacoes = {
            "apoiada": "As fontes encontradas trazem informações que apoiam essa afirmação.",
            "refutada": "As fontes encontradas trazem informações contrárias a essa afirmação.",
            "evidencia_insuficiente": (
                "As fontes encontradas não trazem detalhes suficientes para confirmar ou negar "
                "essa afirmação."
            ),
            "conflitante": "As fontes encontradas discordam entre si sobre essa afirmação.",
            "nao_verificada": (
                "Não conseguimos encontrar informação confiável suficiente para avaliar essa "
                "afirmação."
            ),
        }
        explicacao = explicacoes.get(
            veredito.categoria,
            "O resultado disponível ainda precisa ser interpretado com cuidado.",
        )
        trecho = cls._trecho_encontrado(veredito.justificativa)
        if trecho:
            encerramento = "" if trecho.endswith((".", "!", "?")) else "."
            explicacao += f' Um trecho encontrado diz: “{trecho}”{encerramento}'
        return explicacao

    @staticmethod
    def _resumo_investigacao(estado: EstadoSocratico) -> str:
        contagens: dict[str, int] = {}
        for veredito in estado.vereditos:
            contagens[veredito.categoria] = contagens.get(veredito.categoria, 0) + 1
        apoiadas = contagens.get("apoiada", 0)
        refutadas = contagens.get("refutada", 0)
        incertas = sum(contagens.get(categoria, 0) for categoria in (
            "evidencia_insuficiente", "conflitante", "nao_verificada",
        ))
        partes = []
        if apoiadas:
            partes.append(
                f"{apoiadas} {'afirmações' if apoiadas != 1 else 'afirmação'} com apoio nas fontes"
            )
        if refutadas:
            partes.append(
                f"{refutadas} {'afirmações' if refutadas != 1 else 'afirmação'} "
                f"contrariada{'s' if refutadas != 1 else ''} pelas fontes"
            )
        if incertas:
            partes.append(
                f"{incertas} {'afirmações' if incertas != 1 else 'afirmação'} sem conclusão segura"
            )
        if not partes:
            return "A checagem não conseguiu chegar a uma conclusão segura sobre a notícia."
        return "Na checagem, encontramos " + ", ".join(partes) + "."

    @staticmethod
    def _pergunta_valida(pergunta: str) -> bool:
        return bool(
            10 <= len(pergunta) <= 240
            and pergunta.endswith("?")
            and not re.search(
                r"\b(NLI|score|inferência|metadados?|classificação|veredito|embedding|"
                r"metodologia|viés|fonte primária|fonte secundária)\b",
                pergunta, re.IGNORECASE,
            )
        )

    @staticmethod
    def _fala_continua_no_tema(estado: EstadoSocratico, fala: str) -> bool:
        """Evita que discordâncias e suspeitas sobre a notícia sejam descartadas."""
        def palavras(texto: str) -> set[str]:
            normalizado = unicodedata.normalize("NFKD", texto.lower())
            sem_acentos = "".join(c for c in normalizado if not unicodedata.combining(c))
            return {p for p in re.findall(r"[a-z0-9]+", sem_acentos) if len(p) >= 4}

        palavras_fala = palavras(fala)
        contexto = " ".join(
            [estado.noticia_original] + [v.texto_alegacao for v in estado.vereditos]
        )
        if palavras_fala & palavras(contexto):
            return True
        referencias = {
            "isso", "essa", "esse", "essas", "esses", "deles", "delas", "nisso",
            "noticia", "afirmacao", "fonte", "fontes", "verdade", "mentira",
        }
        if palavras_fala & referencias:
            return True
        # Suspeitas de autoria, intenção ou fraude são relevantes quando surgem
        # como resposta à investigação, mesmo que não estejam comprovadas nela.
        radicais_relevantes = (
            "fraud", "golp", "manipul", "fiscal", "partid", "politic", "govern",
            "eleic", "vot", "urn", "audit", "corrup", "petist", "bolsonar",
        )
        return any(
            palavra.startswith(radical)
            for palavra in palavras_fala
            for radical in radicais_relevantes
        )

    @classmethod
    def _fatos_apoiados_resumidos(cls, estado: EstadoSocratico) -> str:
        fatos = [
            cls._texto_curto(veredito.texto_alegacao, 145)
            for veredito in estado.vereditos
            if veredito.categoria == "apoiada"
        ][:2]
        if not fatos:
            return "As fontes consultadas não oferecem uma base segura para essa conclusão."
        if len(fatos) == 1:
            return f"As fontes consultadas apoiam esta informação: {fatos[0]}."
        return f"As fontes consultadas apoiam que {fatos[0]}; e que {fatos[1]}."

    def _analisar_turno(
        self, estado: EstadoSocratico, resposta_usuario: str, ultimo_turno: bool,
    ) -> dict[str, str]:
        fallback = {
            "relacao": "nao_comprovada_pela_investigacao",
            "resposta": (
                "Essa conclusão não aparece confirmada no material consultado. O que foi checado "
                "permite avaliar a notícia, mas não transforma essa nova ideia em fato."
            ),
            "pergunta": (
                "O que levou você a essa conclusão e que informação poderia fazê-lo reconsiderá-la?"
            ),
        }
        if self.llm is None:
            return fallback

        sistema = (
            "Você conduz um diálogo socrático curto em português simples. A conversa é guiada pelo "
            "que o usuário acabou de dizer, e não pela ordem das alegações. Compare a fala com TODA a "
            "investigação fornecida. Responda diretamente a dúvidas e examine afirmações novas sem "
            "tratá-las como fatos. Se a pessoa atribuir fraude, intenção ou autoria política sem apoio "
            "nos dados, diga com respeito que a investigação não confirma isso e explique o que ela "
            "realmente mostra. Divergir não torna a fala fora de escopo: marque fora_do_escopo somente "
            "para assuntos sem relação com a notícia. A resposta deve ter de uma a três frases e usar "
            "apenas as alegações cujos IDs você citar. A próxima pergunta deve nascer da fala do usuário, "
            "estimular reflexão e não pressupor que a fala seja verdadeira. Não use jargão técnico, não "
            "invente fatos e não avance automaticamente para outra alegação. No último turno, deixe a "
            "pergunta vazia."
        )
        schema = {
            "type": "object",
            "properties": {
                "relacao": {
                    "type": "string",
                    "enum": [
                        "apoiada_pela_investigacao",
                        "contrariada_pela_investigacao",
                        "nao_comprovada_pela_investigacao",
                        "fora_do_escopo",
                    ],
                },
                "ids_alegacoes_usadas": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "uniqueItems": True,
                },
                "resposta": {"type": "string"},
                "pergunta": {"type": "string"},
            },
            "required": ["relacao", "ids_alegacoes_usadas", "resposta", "pergunta"],
            "additionalProperties": False,
        }
        try:
            dados = self.llm.gerar_json(
                sistema,
                json.dumps({
                    **self._contexto(estado),
                    "fala_atual_do_usuario_nao_verificada": resposta_usuario,
                    "este_e_o_ultimo_turno": ultimo_turno,
                }, ensure_ascii=False),
                schema,
                "turno_socratico",
            )
            if set(dados) != {"relacao", "ids_alegacoes_usadas", "resposta", "pergunta"}:
                raise ValueError("Contrato do turno inválido")
            relacao = dados["relacao"]
            if relacao not in {
                "apoiada_pela_investigacao", "contrariada_pela_investigacao",
                "nao_comprovada_pela_investigacao", "fora_do_escopo",
            }:
                raise ValueError("Relação inválida")
            ids_validos = {veredito.alegacao_id for veredito in estado.vereditos}
            ids_usados = dados["ids_alegacoes_usadas"]
            if not isinstance(ids_usados, list) or any(
                not isinstance(item, int) or item not in ids_validos for item in ids_usados
            ):
                raise ValueError("Referência a alegação inválida")
            if relacao in {
                "apoiada_pela_investigacao", "contrariada_pela_investigacao",
            } and estado.vereditos and not ids_usados:
                raise ValueError("Turno relacionado sem fundamentação")
            resposta = " ".join(dados["resposta"].strip().split())
            pergunta = " ".join(dados["pergunta"].strip().split())
            if not 10 <= len(resposta) <= 600:
                raise ValueError("Resposta inválida")
            if ultimo_turno and pergunta:
                raise ValueError("Último turno não pode abrir outra pergunta")
            if not ultimo_turno and not self._pergunta_valida(pergunta):
                raise ValueError("Pergunta inválida")
            if relacao == "fora_do_escopo" and self._fala_continua_no_tema(
                estado, resposta_usuario,
            ):
                relacao = "nao_comprovada_pela_investigacao"
                resposta = (
                    "A investigação não confirma essa conclusão. "
                    f"{self._fatos_apoiados_resumidos(estado)} "
                    "Isso não estabelece a autoria, a intenção ou a suspeita levantada como fato."
                )
                pergunta = (
                    "O que no material levou você a essa interpretação e que informação poderia "
                    "fazê-lo reconsiderá-la?"
                )
            elif relacao == "fora_do_escopo":
                resposta = (
                    "Isso parece fugir do assunto investigado. Podemos voltar à notícia e ao que "
                    "as fontes encontradas permitem avaliar."
                )
            elif relacao == "nao_comprovada_pela_investigacao" and not re.search(
                r"\b(não|sem|insuficiente|incerto|incerteza)\b", resposta, re.IGNORECASE,
            ):
                resposta = "A investigação não confirma essa conclusão. " + resposta
            return {"relacao": relacao, "resposta": resposta, "pergunta": pergunta}
        except Exception:
            return fallback

    def _gerar_sintese_final(self, chave: tuple[int, int], resposta_final: str) -> str:
        estado = self._sessoes.pop(chave)
        linhas = [resposta_final, "", "💡 Síntese da reflexão", "", self._resumo_investigacao(estado)]
        if "nao_comprovada_pela_investigacao" in estado.relacoes_turnos:
            linhas.append(
                "Algumas conclusões levantadas na conversa não estavam confirmadas pelas fontes "
                "consultadas e, por isso, permanecem como hipóteses."
            )
        if "contrariada_pela_investigacao" in estado.relacoes_turnos:
            linhas.append("Parte do que foi levantado entrou em conflito com o material encontrado.")
        if "fora_do_escopo" in estado.relacoes_turnos:
            linhas.append("Um ponto ficou fora do tema e foi redirecionado para a investigação.")
        linhas.extend([
            "",
            "O que encontramos pode mudar se surgirem novas informações. Use esta conversa como "
            "apoio para formar sua própria opinião.",
        ])
        return "\n".join(linhas)
