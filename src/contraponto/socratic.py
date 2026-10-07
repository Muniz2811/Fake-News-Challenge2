"""Módulo para gerenciamento do Diálogo Socrático no chat privado (MVP-12)."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from verifier import VereditoAlegacao


@dataclass
class EstadoSocratico:
    """Armazena o estado mínimo necessário do diálogo por usuário."""
    noticia_original: str
    vereditos: List[VereditoAlegacao]
    persona_modo: str = "Acolhedor"  # 'Acolhedor' (Sônia), 'Dossiê' (Lucas), 'Freio Rápido' (Roberto)
    pergunta_atual_index: int = 0
    respostas_usuario: List[str] = field(default_factory=list)
    perguntas_geradas: List[str] = field(default_factory=list)
    finalizado: bool = False
    expira_em: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=24))


class GerenciadorSocratico:
    """Gere as rodadas do Diálogo Socrático (máximo 3 perguntas) e a síntese final."""

    MAX_PERGUNTAS = 3

    def __init__(self):
        self._sessoes: Dict[int, EstadoSocratico] = {}

    def _limpar_expiradas(self) -> None:
        agora = datetime.now()
        expirados = [uid for uid, estado in self._sessoes.items() if estado.expira_em <= agora]
        for uid in expirados:
            del self._sessoes[uid]

    def iniciar_sessao(
        self, user_id: int, noticia_original: str, vereditos: List[VereditoAlegacao], persona_modo: str = "Acolhedor"
    ) -> str:
        """Inicia uma nova jornada socrática para o usuário adaptada por persona."""
        self._limpar_expiradas()
        perguntas = self._gerar_perguntas_iniciais(noticia_original, vereditos, persona_modo)

        estado = EstadoSocratico(
            noticia_original=noticia_original,
            vereditos=vereditos,
            persona_modo=persona_modo,
            pergunta_atual_index=0,
            perguntas_geradas=perguntas
        )
        self._sessoes[user_id] = estado

        primeira_pergunta = perguntas[0]

        header = "🏛️ **Diálogo Socrático (Contra)**\n"
        if persona_modo == "Acolhedor":
            header += "Oi! Vamos analisar juntas(os) esta mensagem em até 3 perguntas simples?\n"
        elif persona_modo == "Freio Rápido":
            header += "Análise rápida de contexto em 3 perguntas diretas.\n"
        else:
            header += "Investigação socrática de fontes e evidências (máx. 3 rodadas).\n"

        return (
            f"{header}"
            "_(Envie /cancelar a qualquer momento para parar)._\n\n"
            f"**Pergunta 1/3:** {primeira_pergunta}"
        )

    def processar_resposta(self, user_id: int, resposta_texto: str) -> str:
        """Processa a resposta do usuário considerando a entrada anterior e avança o diálogo."""
        self._limpar_expiradas()
        if user_id not in self._sessoes or self._sessoes[user_id].finalizado:
            return "Não há nenhuma sessão ativa do Diálogo Socrático. Envie uma mensagem para iniciar!"

        estado = self._sessoes[user_id]

        # Registra a resposta do usuário
        resposta_limpa = resposta_texto.strip()
        estado.respostas_usuario.append(resposta_limpa)
        estado.pergunta_atual_index += 1

        # Atingiu o limite de 3 perguntas
        if estado.pergunta_atual_index >= self.MAX_PERGUNTAS or estado.pergunta_atual_index >= len(estado.perguntas_geradas):
            return self._gerar_sintese_final(user_id)

        # Prepara a próxima pergunta considerando a resposta que o usuário acabou de enviar
        proxima_pergunta = estado.perguntas_geradas[estado.pergunta_atual_index]
        num_p = estado.pergunta_atual_index + 1

        feedback = f" Entendi o seu ponto quando disse: *'{resposta_limpa}'*.\n\n"
        return f"{feedback}**Pergunta {num_p}/3:** {proxima_pergunta}"

    def cancelar_sessao(self, user_id: int) -> str:
        """Permite cancelar o diálogo socrático voluntariamente."""
        self._limpar_expiradas()
        if user_id in self._sessoes:
            del self._sessoes[user_id]
            return "🚪 Diálogo Socrático encerrado. Fique à vontade para consultar novas notícias sempre que precisar!"
        return "Nenhuma sessão ativa para encerrar."

    def tem_sessao_ativa(self, user_id: int) -> bool:
        self._limpar_expiradas()
        return user_id in self._sessoes and not self._sessoes[user_id].finalizado

    def _gerar_perguntas_iniciais(
        self, noticia: str, vereditos: List[VereditoAlegacao], persona_modo: str
    ) -> List[str]:
        """Gere perguntas adaptadas com base nos vereditos e na persona."""
        alegacao_txt = vereditos[0].texto_alegacao if vereditos else noticia

        if persona_modo == "Acolhedor":  # Dona Sônia
            return [
                f"Ao ler '{alegacao_txt}', o que mais te chamou atenção ou trouxe preocupação?",
                "Você notou se essa mensagem foi enviada por algum órgão oficial ou se apenas pedia para 'repassar urgente'?",
                "Se você estivesse em dúvida sobre esse assunto, que pessoa ou site de confiança você procuraria antes de compartilhar?"
            ]
        elif persona_modo == "Freio Rápido":  # Roberto
            return [
                f"A afirmação '{alegacao_txt}' possui link direto para a fonte primária?",
                "Qual detalhe no texto pareceu omitido ou fora de proporção?",
                "Considerando o risco de desinformação, vale a pena aguardar a confirmação oficial antes de repassar?"
            ]
        else:  # Lucas (Modo Dossiê)
            return [
                f"Em relação a '{alegacao_txt}', quais evidências concretas ou dados o texto apresenta?",
                "Comparando com checagens registradas em bases públicas, qual inconsistência você identifica no contexto?",
                "Quais elementos seriam necessários para que você considerasse essa afirmação 100% comprovada?"
            ]

    def _gerar_sintese_final(self, user_id: int) -> str:
        """Sintetiza as evidências e o diálogo com foco no reconhecimento de incertezas."""
        estado = self._sessoes[user_id]
        estado.finalizado = True

        evidencias_resumo = []
        for v in estado.vereditos:
            status_formatado = v.categoria.replace('_', ' ').title()
            evidencias_resumo.append(f"• **Afirmação:** '{v.texto_alegacao}'")
            evidencias_resumo.append(f"   - **Situação de Checagem:** {status_formatado}")
            if v.fontes_citadas:
                for f in v.fontes_citadas:
                    evidencias_resumo.append(f"   - **Fonte:** {f}")

        resumo_str = "\n".join(evidencias_resumo) if evidencias_resumo else "• Sem evidências conclusivas registradas até o momento."

        sintese = (
            "💡 **Síntese da Investigação (Divã Socrático)**\n\n"
            "Obrigado por dedicar este tempo para refletir criticamente sobre o conteúdo!\n\n"
            "📋 **Resumo das Evidências Encontradas:**\n"
            f"{resumo_str}\n\n"
            "⚖️ **Reconhecimento de Incertezas & Autonomia:**\n"
            "As checagens refletem as evidências públicas disponíveis na web até o momento. "
            "A informação pode mudar conforme novos dados surjam. A decisão final sobre a veracidade e o compartilhamento é sempre sua.\n\n"
            "_(Sessão finalizada. Envie uma nova mensagem sempre que desejar investigar!)_"
        )

        del self._sessoes[user_id]
        return sintese
