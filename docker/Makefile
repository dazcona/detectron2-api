build:
	@docker-compose -f docker-compose.yml build
run:
	@docker-compose -f docker-compose.yml up -d
dev:
	@docker exec -it dazcona_express_human-mask-api bash
status:
	@docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
down:
	@docker-compose -f docker-compose.yml down -v
reset:
	@docker stop $(docker ps -q); docker rm $(docker ps -a -q)
prune:
	@docker network prune -f; docker volume prune -f
kill:
	make down; make reset; make prune; make run