#include <stdio.h>
#include <cstdlib>
#include "csapp.h"
#include <cstring>

int main(int argc, char *argv[])
{
	int listenfd, connfd;
	char hostname[MAXLINE], port[MAXLINE];
	socklen_t clientlen;
	struct sockaddr_storage clientaddr;
	int acc_count = 0;

	//if (strcmp(argv[2], "compute-0-29.local") == 0)
	//{
	printf("%s listening\n", argv[2]);
	listenfd = Open_listenfd ("15618");
	//}
	sleep(5);

	//int send_val = 5;
	printf("Hostname is %s\n", argv[2]);
	printf("Int got as %d\n", atoi(argv[1]));

	if (strcmp(argv[2], "compute-0-29.local") == 0)
	{
		while (1)
		{
			clientlen = sizeof (clientaddr);

			// accept connections 
			connfd = Accept (listenfd, (SA *) & clientaddr, &clientlen);
			Getnameinfo ((SA *) & clientaddr, clientlen, hostname, MAXLINE,
					port, MAXLINE, 0);
			printf ("Accepted connection from (%s, %s). Connfd is %d\n", hostname, port, connfd);

			//newfd = (int *) malloc (sizeof (int));
			//newfd = connfd;

			// go serve this client! 
			// pthread_create (&tid, NULL, doit, newfd);
			acc_count++;

			double send_double = 232.23;
			int retval = Rio_writen (connfd, (void *)&send_double, sizeof(double));
			if (retval < 0)
			{   
				printf("Rio_writen to %d encountered a problem.\n", connfd);

				unix_error ("Rio_writen error");
			}   

			retval = Rio_writen (connfd, (void *)&send_double, sizeof(double));
			if (retval < 0)
			{   
				printf("Rio_writen to %d encountered a problem.\n", connfd);

				unix_error ("Rio_writen error");
			}
			int len = Rio_readn (connfd, (void *)&send_double, sizeof(double));

			if (len < 0)
			{
				unix_error ("Rio_readlineb error");
			}
			printf("Host %s got len as %d and receive_val as %lf\n", argv[2], len, send_double);

			len = Rio_readn (connfd, (void *)&send_double, sizeof(double));

			if (len < 0)
			{
				unix_error ("Rio_readlineb error");
			}
			printf("Host %s got len as %d and receive_val as %lf\n", argv[2], len, send_double);

			if (acc_count == 3)
			{
				printf("Accepted 3 connections.\n");
				break;
			}
		}
	}
	else
	{
		int serverfd = Open_clientfd ("10.22.1.241", "15618");
		printf("In host %s, serverfd is %d\n", argv[2], serverfd);

		double buf;
		int len = Rio_readn (serverfd, (void *)&buf, sizeof(double));

		if (len < 0)
		{
			unix_error ("Rio_readlineb error");
		}
		printf("Host %s got len as %d and buf as %lf\n", argv[2], len, buf);

		len = Rio_readn (serverfd, (void *)&buf, sizeof(double));

		if (len < 0)
		{
			unix_error ("Rio_readlineb error");
		}
		printf("Host %s got len as %d and buf as %lf\n", argv[2], len, buf);

		buf = 99.104;
		int retval = Rio_writen (serverfd, (void *)&buf, sizeof(double));
		if (retval < 0)
		{   
			printf("Rio_writen to %d encountered a problem.\n", serverfd);

			unix_error ("Rio_writen error");
		}   

		retval = Rio_writen (serverfd, (void *)&buf, sizeof(double));

		if (retval < 0)
		{   
			printf("Rio_writen to %d encountered a problem.\n", serverfd);

			unix_error ("Rio_writen error");
		}   
	}
	return 0;
}
