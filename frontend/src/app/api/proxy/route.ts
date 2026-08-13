import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const { url } = await request.json();
    
    // We proxy the request through the Next.js server to avoid CORS preflight errors from Localtunnel
    const res = await fetch(url, {
      headers: {
        'bypass-tunnel-reminder': 'true',
        'User-Agent': 'macro-forecaster-dashboard'
      },
      cache: 'no-store'
    });
    
    if (!res.ok) {
      throw new Error(`Backend returned ${res.status} ${res.statusText}`);
    }
    
    const data = await res.json();
    return NextResponse.json(data);
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
