import { NextResponse } from 'next/server';

const BACKEND_URL = "http://192.248.43.132:8085";

export async function GET() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/calendar`, {
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
