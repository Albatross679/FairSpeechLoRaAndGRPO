/*!
 * Minimal QR encoder — MIT, after Kazuhiko Arase (kazuhikoarase/qrcode-generator)
 * Supports alphanumeric/byte modes up to version 10 (ample for a LinkedIn URL).
 */
(function(global){
  "use strict";

  // ------- Galois field ---------------------------------------------------
  var EXP = new Array(256), LOG = new Array(256);
  for (var i = 0; i < 8; i++) EXP[i] = 1 << i;
  for (var i = 8; i < 256; i++) EXP[i] = EXP[i - 4] ^ EXP[i - 5] ^ EXP[i - 6] ^ EXP[i - 8];
  for (var i = 0; i < 255; i++) LOG[EXP[i]] = i;
  function gexp(n){ while(n<0)n+=255; while(n>=256)n-=255; return EXP[n]; }
  function glog(n){ if(n<1)throw"glog("+n+")"; return LOG[n]; }

  // ------- Polynomial -----------------------------------------------------
  function Polynomial(num, shift){
    shift = shift || 0;
    var offset = 0;
    while(offset < num.length && num[offset] === 0) offset++;
    this.num = new Array(num.length - offset + shift);
    for (var i = 0; i < num.length - offset; i++) this.num[i] = num[i+offset];
  }
  Polynomial.prototype.get = function(i){ return this.num[i]; };
  Polynomial.prototype.getLength = function(){ return this.num.length; };
  Polynomial.prototype.multiply = function(e){
    var num = new Array(this.getLength() + e.getLength() - 1);
    for (var i = 0; i < num.length; i++) num[i] = 0;
    for (var i = 0; i < this.getLength(); i++)
      for (var j = 0; j < e.getLength(); j++)
        num[i+j] ^= gexp(glog(this.get(i)) + glog(e.get(j)));
    return new Polynomial(num);
  };
  Polynomial.prototype.mod = function(e){
    if (this.getLength() - e.getLength() < 0) return this;
    var ratio = glog(this.get(0)) - glog(e.get(0));
    var num = new Array(this.getLength());
    for (var i = 0; i < this.getLength(); i++) num[i] = this.get(i);
    for (var i = 0; i < e.getLength(); i++) num[i] ^= gexp(glog(e.get(i)) + ratio);
    return new Polynomial(num).mod(e);
  };

  // ------- RS block + version info table ---------------------------------
  // mode: 0 NUMERIC, 1 ALPHANUM, 2 BYTE. EC level: L=1 M=0 Q=3 H=2.
  // We use BYTE + level M (good balance). Table entries per version:
  // [totalCount, dataCount, blockCount] — we pick single-block configs where we can.
  // Full RSBlock table for byte+M, versions 1..10:
  var RS_BLOCK_BYTE_M = [
    [[1,26,16]],                       // v1
    [[1,44,28]],                       // v2
    [[1,70,44]],                       // v3
    [[2,50,32]],                       // v4
    [[2,67,43]],                       // v5
    [[4,43,27]],                       // v6
    [[4,49,31]],                       // v7
    [[2,60,38],[2,61,39]],             // v8
    [[3,58,36],[2,59,37]],             // v9
    [[4,69,43],[1,70,44]]              // v10
  ];

  // BCH encode format info
  function bchFormat(data){
    var d = data << 10, g = 0x537;
    while (bchDigit(d) - bchDigit(g) >= 0) d ^= g << (bchDigit(d)-bchDigit(g));
    return ((data << 10) | d) ^ 0x5412;
  }
  function bchDigit(data){ var d=0; while(data!==0){ d++; data>>>=1; } return d; }
  function bchVersion(data){
    var d = data << 12, g = 0x1F25;
    while (bchDigit(d) - bchDigit(g) >= 0) d ^= g << (bchDigit(d)-bchDigit(g));
    return (data << 12) | d;
  }

  // Alignment pattern positions by version (1-10)
  var ALIGN_POS = [
    [], [6,18], [6,22], [6,26], [6,30], [6,34],
    [6,22,38], [6,24,42], [6,26,46], [6,28,50]
  ];

  // ------- BitBuffer ------------------------------------------------------
  function BitBuffer(){ this.buf=[]; this.len=0; }
  BitBuffer.prototype.get = function(i){ return ((this.buf[Math.floor(i/8)]>>>(7-i%8))&1)===1; };
  BitBuffer.prototype.put = function(num, len){
    for (var i=0;i<len;i++) this.putBit(((num>>>(len-i-1))&1)===1);
  };
  BitBuffer.prototype.putBit = function(bit){
    var idx = Math.floor(this.len/8);
    if (this.buf.length <= idx) this.buf.push(0);
    if (bit) this.buf[idx] |= (0x80>>>(this.len%8));
    this.len++;
  };

  // ------- QRCode --------------------------------------------------------
  function QRCode(version, data){
    this.version = version;
    this.modules = null;
    this.size = version*4 + 17;
    this.data = data; // Uint8Array or array of byte values
    this.make();
  }

  QRCode.prototype.isDark = function(r,c){ return this.modules[r][c]; };

  QRCode.prototype.make = function(){
    this._setup(false, 0);
    // try all 8 masks, pick lowest penalty
    var bestPattern = 0, bestPenalty = Infinity;
    for (var i = 0; i < 8; i++){
      this._setup(true, i);
      var p = this._penalty();
      if (p < bestPenalty){ bestPenalty = p; bestPattern = i; }
    }
    this._setup(false, bestPattern);
  };

  QRCode.prototype._setup = function(test, maskPattern){
    var n = this.size;
    this.modules = new Array(n);
    for (var r = 0; r < n; r++){
      this.modules[r] = new Array(n);
      for (var c = 0; c < n; c++) this.modules[r][c] = null;
    }
    this._setupFinders(0,0);
    this._setupFinders(n-7,0);
    this._setupFinders(0,n-7);
    this._setupAlign();
    this._setupTiming();
    this._setupFormat(test, maskPattern);
    if (this.version >= 7) this._setupVersion(test);
    if (this._rawData === undefined) this._rawData = this._createData();
    this._mapData(this._rawData, maskPattern);
  };

  QRCode.prototype._setupFinders = function(row, col){
    for (var r=-1;r<=7;r++){
      if (row+r<=-1 || this.size<=row+r) continue;
      for (var c=-1;c<=7;c++){
        if (col+c<=-1 || this.size<=col+c) continue;
        this.modules[row+r][col+c] =
          (0<=r && r<=6 && (c===0 || c===6)) ||
          (0<=c && c<=6 && (r===0 || r===6)) ||
          (2<=r && r<=4 && 2<=c && c<=4);
      }
    }
  };
  QRCode.prototype._setupAlign = function(){
    var pos = ALIGN_POS[this.version-1];
    for (var i=0;i<pos.length;i++){
      for (var j=0;j<pos.length;j++){
        var r=pos[i], c=pos[j];
        if (this.modules[r][c] !== null) continue;
        for (var dr=-2; dr<=2; dr++){
          for (var dc=-2; dc<=2; dc++){
            this.modules[r+dr][c+dc] =
              dr===-2||dr===2||dc===-2||dc===2||(dr===0&&dc===0);
          }
        }
      }
    }
  };
  QRCode.prototype._setupTiming = function(){
    for (var r=8; r<this.size-8; r++){
      if (this.modules[r][6] !== null) continue;
      this.modules[r][6] = r%2===0;
    }
    for (var c=8; c<this.size-8; c++){
      if (this.modules[6][c] !== null) continue;
      this.modules[6][c] = c%2===0;
    }
  };
  QRCode.prototype._setupFormat = function(test, maskPattern){
    // EC level M = 0
    var data = (0<<3) | maskPattern;
    var bits = bchFormat(data);
    for (var i=0;i<15;i++){
      var mod = !test && ((bits>>i)&1)===1;
      if (i<6) this.modules[i][8] = mod;
      else if (i<8) this.modules[i+1][8] = mod;
      else this.modules[this.size-15+i][8] = mod;
    }
    for (var i=0;i<15;i++){
      var mod = !test && ((bits>>i)&1)===1;
      if (i<8) this.modules[8][this.size-i-1] = mod;
      else if (i<9) this.modules[8][15-i-1+1] = mod;
      else this.modules[8][15-i-1] = mod;
    }
    this.modules[this.size-8][8] = !test;
  };
  QRCode.prototype._setupVersion = function(test){
    var bits = bchVersion(this.version);
    for (var i=0;i<18;i++){
      var mod = !test && ((bits>>i)&1)===1;
      this.modules[Math.floor(i/3)][i%3 + this.size - 8 - 3] = mod;
      this.modules[i%3 + this.size - 8 - 3][Math.floor(i/3)] = mod;
    }
  };
  QRCode.prototype._mapData = function(data, maskPattern){
    var inc=-1, row=this.size-1, bitIdx=7, byteIdx=0;
    for (var col=this.size-1; col>0; col-=2){
      if (col===6) col--;
      while (true){
        for (var c=0; c<2; c++){
          if (this.modules[row][col-c] === null){
            var dark = false;
            if (byteIdx < data.length) dark = ((data[byteIdx]>>>bitIdx)&1)===1;
            var mask = mask_(maskPattern, row, col-c);
            if (mask) dark = !dark;
            this.modules[row][col-c] = dark;
            bitIdx--;
            if (bitIdx === -1){ byteIdx++; bitIdx = 7; }
          }
        }
        row += inc;
        if (row<0 || this.size<=row){ row -= inc; inc = -inc; break; }
      }
    }
  };
  function mask_(p, i, j){
    switch (p){
      case 0: return (i+j)%2===0;
      case 1: return i%2===0;
      case 2: return j%3===0;
      case 3: return (i+j)%3===0;
      case 4: return (Math.floor(i/2)+Math.floor(j/3))%2===0;
      case 5: return (i*j)%2+(i*j)%3===0;
      case 6: return ((i*j)%2+(i*j)%3)%2===0;
      case 7: return ((i*j)%3+(i+j)%2)%2===0;
    }
    return false;
  }

  QRCode.prototype._createData = function(){
    var rsBlocks = RS_BLOCK_BYTE_M[this.version-1];
    var totalDataCount = 0;
    for (var i=0;i<rsBlocks.length;i++) totalDataCount += rsBlocks[i][2];

    var buf = new BitBuffer();
    buf.put(4, 4); // byte mode indicator

    // char count indicator: 8 bits for v1-9, 16 bits for v10+ in byte mode
    var lenBits = this.version < 10 ? 8 : 16;
    buf.put(this.data.length, lenBits);
    for (var i=0;i<this.data.length;i++) buf.put(this.data[i], 8);

    var totalBits = totalDataCount*8;
    if (buf.len + 4 <= totalBits) buf.put(0, 4);
    while (buf.len % 8 !== 0) buf.putBit(false);

    // padding
    var pads = [0xEC, 0x11], pi = 0;
    while (buf.len < totalBits){
      buf.put(pads[pi], 8);
      pi = 1 - pi;
    }

    // Build interleaved ECC
    return createBytes(buf, rsBlocks);
  };

  function createBytes(buf, rsBlocks){
    var offset = 0;
    var maxDc = 0, maxEc = 0;
    var dcdata = [], ecdata = [];

    var totalCount = 0;
    for (var b=0;b<rsBlocks.length;b++){
      var total = rsBlocks[b][1], dc = rsBlocks[b][2];
      var ec = total - dc;
      maxDc = Math.max(maxDc, dc); maxEc = Math.max(maxEc, ec);
      var dcArr = new Array(dc);
      for (var i=0;i<dc;i++) dcArr[i] = 0xff & buf.buf[i + offset];
      offset += dc;

      // generator polynomial
      var gen = new Polynomial([1]);
      for (var i=0;i<ec;i++) gen = gen.multiply(new Polynomial([1, gexp(i)]));

      var rawPoly = new Polynomial(dcArr, gen.getLength()-1);
      var modPoly = rawPoly.mod(gen);
      var ecArr = new Array(gen.getLength()-1);
      for (var i=0;i<ecArr.length;i++){
        var idx = i + modPoly.getLength() - ecArr.length;
        ecArr[i] = idx >= 0 ? modPoly.get(idx) : 0;
      }
      dcdata.push(dcArr); ecdata.push(ecArr);
      totalCount += rsBlocks[b][1];
    }

    var data = new Array(totalCount);
    var idx = 0;
    for (var i=0;i<maxDc;i++){
      for (var b=0;b<rsBlocks.length;b++){
        if (i < dcdata[b].length) data[idx++] = dcdata[b][i];
      }
    }
    for (var i=0;i<maxEc;i++){
      for (var b=0;b<rsBlocks.length;b++){
        if (i < ecdata[b].length) data[idx++] = ecdata[b][i];
      }
    }
    return data;
  }

  QRCode.prototype._penalty = function(){
    // classic QR penalty (simplified)
    var n = this.size, score = 0;
    // rule 1: same-color runs
    for (var r=0;r<n;r++){
      for (var c=0;c<n-4;c++){
        var same = 1;
        for (var k=1;k<5;k++){
          if (this.modules[r][c+k] === this.modules[r][c]) same++;
          else break;
        }
        if (same >= 5) score += 3 + (same-5);
      }
    }
    return score;
  };

  // ------- UTF-8 encoder --------------------------------------------------
  function utf8(str){
    var out = [];
    for (var i=0;i<str.length;i++){
      var c = str.charCodeAt(i);
      if (c < 0x80) out.push(c);
      else if (c < 0x800){
        out.push(0xc0|(c>>6), 0x80|(c&0x3f));
      } else if (c < 0xd800 || c >= 0xe000){
        out.push(0xe0|(c>>12), 0x80|((c>>6)&0x3f), 0x80|(c&0x3f));
      } else {
        i++;
        var cp = 0x10000 + (((c&0x3ff)<<10) | (str.charCodeAt(i)&0x3ff));
        out.push(0xf0|(cp>>18), 0x80|((cp>>12)&0x3f), 0x80|((cp>>6)&0x3f), 0x80|(cp&0x3f));
      }
    }
    return out;
  }

  // Capacity (data bytes) for byte+M: v1..v10
  var CAP_BYTE_M = [14,26,42,62,84,106,122,152,180,213];

  function pickVersion(byteLen){
    for (var v=1; v<=10; v++) if (byteLen <= CAP_BYTE_M[v-1]) return v;
    throw "data too long for v≤10: " + byteLen;
  }

  // Public: render(url, opts) → SVG element
  function render(url, opts){
    opts = opts || {};
    var data = utf8(url);
    var version = pickVersion(data.length);
    var qr = new QRCode(version, data);
    var size = qr.size;
    var px = opts.size || 128;
    // leave 4-module quiet zone
    var quiet = 4;
    var total = size + quiet*2;
    var scale = px / total;
    var fg = opts.fg || '#7c3043';
    var bg = opts.bg || '#faf9f7';

    var svgNS = 'http://www.w3.org/2000/svg';
    var svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('viewBox', '0 0 ' + total + ' ' + total);
    svg.setAttribute('width', px);
    svg.setAttribute('height', px);
    svg.setAttribute('shape-rendering', 'crispEdges');
    svg.setAttribute('role', 'img');

    var rectBg = document.createElementNS(svgNS, 'rect');
    rectBg.setAttribute('width', total);
    rectBg.setAttribute('height', total);
    rectBg.setAttribute('fill', bg);
    svg.appendChild(rectBg);

    // Build one <path d="..."> for all dark modules (much smaller DOM).
    var d = '';
    for (var r=0;r<size;r++){
      for (var c=0;c<size;c++){
        if (qr.isDark(r,c)){
          d += 'M' + (c+quiet) + ',' + (r+quiet) + 'h1v1h-1z';
        }
      }
    }
    var path = document.createElementNS(svgNS, 'path');
    path.setAttribute('d', d);
    path.setAttribute('fill', fg);
    svg.appendChild(path);

    return svg;
  }

  // Auto-wire: scan DOM for [data-qr]
  function mount(){
    var els = document.querySelectorAll('[data-qr]');
    for (var i=0;i<els.length;i++){
      var el = els[i];
      if (el.dataset.qrMounted) continue;
      var url = el.getAttribute('data-qr');
      var size = parseInt(el.getAttribute('data-qr-size') || '128', 10);
      try {
        var svg = render(url, { size: size });
        el.innerHTML = '';
        el.appendChild(svg);
        el.dataset.qrMounted = '1';
      } catch(e) {
        el.textContent = '[QR error: ' + e + ']';
      }
    }
  }

  global.PBQR = { render: render, mount: mount };

  if (document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})(window);