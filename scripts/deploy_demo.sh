#!/bin/bash

case "$1" in
  start)
    echo "🚀 Iniciando ambiente de demonstração do bot..."
    docker compose up -d --build
    ;;
  stop)
    echo "🛑 Parando ambiente de demonstração..."
    docker compose down
    ;;
  restart)
    echo "🔄 Reiniciando ambiente de demonstração..."
    docker compose restart
    ;;
  logs)
    echo "📜 Exibindo logs em tempo real..."
    docker compose logs -f
    ;;
  *)
    echo "Uso: $0 {start|stop|restart|logs}"
    exit 1
    ;;
esac