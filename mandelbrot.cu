__host__ __device__ void calcMandelbrotPixel( unsigned char *img, unsigned int x, unsigned int y )
{
	double x0, y0, xb, yb, xtmp;
	unsigned int i, iter;
	
	if( x < WIDTH && y < HEIGHT ) {
		i = ( x + y * WIDTH ) * 3;
		
		x0 = ( (double)x / WIDTH * 3.5 ) - 2.5;
		y0 = ( (double)y / HEIGHT * 2 ) - 1;

		xb = 0;
		yb = 0;

		iter = 0;

		while( xb*xb + yb*yb < 4 && iter < MAX_ITER ) {
			xtmp = xb*xb - yb*yb + x0;
			yb = 2*xb*yb + y0;

			xb = xtmp;
			iter++;
		}
				
		iter*=20;
		
		img[ i ] = iter > 510 ? ( iter - 510 ) % 255 : 0;
		img[ i + 1 ] = iter > 255 ? ( iter - 255 ) % 255 : 0;
		img[ i + 2 ] = iter % 255;
	}
}